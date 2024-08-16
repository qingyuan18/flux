import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline

###neuron relevant lib
import copy
import forward_decorator as fd
import torch_neuronx
import torch_xla
import torch_xla.core.xla_model as xm
from pathlib import Path
xla_device = xm.xla_device()


RUN_ON_NEURON=True
NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


name = "flux-schnell"
width = 1360
height = 768
seed = None
prompt = """a photo of a forest with mist swirling around the tree trunks. The word
        'FLUX' is painted over it in big, red brush strokes with visible texture
        """
device = "cuda" if torch.cuda.is_available() else "cpu"
num_steps = None
loop = False
guidance = 3.5
offload = False
output_dir = "output"
add_sampling_metadata = True


nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

if name not in configs:
    available = ", ".join(configs.keys())
    raise ValueError(f"Got unknown model name: {name}, chose from {available}")

torch_device = torch.device(device)
if num_steps is None:
    num_steps = 4 if name == "flux-schnell" else 50

# allow for packing and conversion to latent space
height = 16 * (height // 16)
width = 16 * (width // 16)

output_name = os.path.join(output_dir, "img_{idx}.jpg")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    idx = 0
else:
    fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]\.jpg$", fn)]
    if len(fns) > 0:
        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
    else:
        idx = 0

# init all components
clip = load_clip(torch_device)
#clip_tokenizer_neuron=fd.make_forward_verbose(model=clip.tokenizer, model_name="openai clip vit tokenizer")
clip_hf_module_neuron=fd.make_forward_verbose(model=clip.hf_module, model_name="openai clip vit hf_module")

model = load_flow_model(name, device="cpu" if offload else torch_device)
model_neuron=fd.make_forward_verbose(model=model, model_name="flux model")

ae = load_ae(name, device="cpu" if offload else torch_device)
ae_neuron=fd.make_forward_verbose(model=ae.decoder, model_name="vae")

## 模型编译
if RUN_ON_NEURON:
    ## 1:模型编译路径
    NEURON_COMPILER_WORKDIR = Path("neuron_compiler_workdir")
    NEURON_COMPILER_WORKDIR.mkdir(exist_ok=True)
    NEURON_COMPILER_OUTPUT_DIR = Path("compiled_models")
    NEURON_COMPILER_OUTPUT_DIR.mkdir(exist_ok=True)

    ## 2: 模型编译参数
    NEURON_COMPILER_TYPE_CASTING_CONFIG = [
    "--auto-cast=matmult",
    f"--auto-cast-type=bf16"
    ]
    NEURON_COMPILER_CLI_ARGS = [
    "--target=inf2",
    "--enable-fast-loading-neuron-binaries",
    *NEURON_COMPILER_TYPE_CASTING_CONFIG,
    ]
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"

    ################# 3.1 vit clip compile ##################

    ## clip tokenizer
    #VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_TOKENIZER"
    #VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR.mkdir(exist_ok=True)
    #example_clip_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
    #with torch.no_grad():
    #   clip_model_neuron = torch_neuronx.trace(
    #       clip_neuron,
    #       example_clip_model_input,
    #           compiler_workdir=VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR,
    #           compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR}/log-neuron-cc.txt'],
    #           )

    #neuron_model=clip_model_neuron
    #file_name ="clip_neuron_tokenizer_model.pt"
    #torch_neuronx.async_load(neuron_model)
    #torch_neuronx.lazy_load(neuron_model)
    #torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)


    ## clip HF model
    VIT_CLIP_HF_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_HF"
    VIT_CLIP_HF_COMPILATION_DIR.mkdir(exist_ok=True)
    # 构建与 kwargs 格式一致的 example input
    example_input_ids = torch.randint(0, 50000, (1, 77), dtype=torch.long)  # 与 kwarg0 格式一致
    example_attention_mask = torch.randint(0, 2, (1,))# 与原始 kwarg1 保持一致 None
    example_output_hidden_states = torch.tensor([False], dtype=torch.bool)

    # 将这些组合成一个元组作为 example input
    example_clip_model_input = (example_input_ids, example_attention_mask, example_output_hidden_states)

    with torch.no_grad():
        clip_model_neuron = torch_neuronx.trace(
               clip_hf_module_neuron,
               example_clip_model_input,
                   compiler_workdir=VIT_CLIP_HF_COMPILATION_DIR,
                   compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VIT_CLIP_HF_COMPILATION_DIR}/log-neuron-cc.txt'],
                   )

        neuron_model=clip_model_neuron
        file_name ="clip_neuron_hf_model.pt"
        torch_neuronx.async_load(neuron_model)
        torch_neuronx.lazy_load(neuron_model)
        torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)

    # Free up memory
    print(neuron_model.code)
    del neuron_model, example_clip_model_input

    ################# 3.2 flux model compile ##################
    FLUX_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "FLUX"
    FLUX_COMPILATION_DIR.mkdir(exist_ok=True)


    example_flux_model_input = (
        torch.randn(1, 1800, 64),  # img: Tensor(1, 1800, 64)
        torch.randint(0, 1000, (1, 1800, 3), dtype=torch.long),  # img_ids: Tensor(1, 1800, 3)
        torch.randn(1, 256, 4096),  # txt: Tensor(1, 256, 4096)
        torch.randint(0, 1000, (1, 256, 3), dtype=torch.long),  # txt_ids: Tensor(1, 256, 3)
        torch.randn(1, 768),  # y: Tensor(1, 768)
        torch.randint(0, 1000, (1,), dtype=torch.long),  # timesteps: Tensor(1,)
        torch.randn(1)  # guidance: Tensor(1,)
    )

    with torch.no_grad():
       flux_model_neuron = torch_neuronx.trace(
            model_neuron,
            example_flux_model_input,
            compiler_workdir=FLUX_COMPILATION_DIR,
            compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={FLUX_COMPILATION_DIR}/log-neuron-cc.txt'],
            )

       neuron_model=flux_model_neuron
       file_name ="flux_neuron_model.pt"
       torch_neuronx.async_load(neuron_model)
       torch_neuronx.lazy_load(neuron_model)
       torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)
         # Free up memory
    print(neuron_model.code)
    del neuron_model, example_flux_model_input

################# 3.3 ae model compile ##################
    AE_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "AE"
    AE_COMPILATION_DIR.mkdir(exist_ok=True)

    example_ae_model_input = torch.randn((1, 16, 90, 80), dtype=DTYPE)
    with torch.no_grad():
        ae_model_neuron = torch_neuronx.trace(
          ae_neuron,
          example_ae_model_input,
          compiler_workdir=AE_COMPILATION_DIR,
          compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={AE_COMPILATION_DIR}/log-neuron-cc.txt'],
          )

        neuron_model=ae_model_neuron
        file_name ="ae_neuron_model.pt"
        torch_neuronx.async_load(neuron_model)
        torch_neuronx.lazy_load(neuron_model)
        torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)
        # Free up memory
        print(neuron_model.code)
        del neuron_model, example_flux_model_input

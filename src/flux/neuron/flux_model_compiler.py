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
import neuron.forward_decorator as fd
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


name: str = "flux-schnell",
width: int = 1360,
height: int = 768,
seed: int | None = None,
prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
),
device: str = "cuda" if torch.cuda.is_available() else "cpu",
num_steps: int | None = None,
loop: bool = False,
guidance: float = 3.5,
offload: bool = False,
output_dir: str = "output",
add_sampling_metadata: bool = True


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
clip_tokenizer_neuron=fd.make_forward_verbose(model=clip.tokenizer, model_name="openai clip vit tokenizer")
clip_hf_module_neuron=fd.make_forward_verbose(model=clip.hf_module, model_name="openai clip vit hf_module")

model = load_flow_model(name, device="cpu" if offload else torch_device)
model_neuron=fd.make_forward_verbose(model=model, model_name="flux model")

ae = load_ae(name, device="cpu" if offload else torch_device)
ae_neuron=fd.make_forward_verbose(model=ae, model_name="vae")

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
    VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_TOKENIZER"
    VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR.mkdir(exist_ok=True)
    example_clip_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
    with torch.no_grad():
       clip_model_neuron = torch_neuronx.trace(
           clip_neuron,
           example_clip_model_input,
               compiler_workdir=VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR,
               compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VIT_CLIP_CLIP_TOKENIZER_COMPILATION_DIR}/log-neuron-cc.txt'],
               )

    neuron_model=clip_model_neuron
    file_name ="clip_neuron_tokenizer_model.pt":
        torch_neuronx.async_load(neuron_model)
        torch_neuronx.lazy_load(neuron_model)
        torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)


    ## clip HF model
    VIT_CLIP_HF_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "CLIP_HF"
    VIT_CLIP_HF_COMPILATION_DIR.mkdir(exist_ok=True)
    example_clip_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
        with torch.no_grad():
           clip_model_neuron = torch_neuronx.trace(
               clip_neuron,
               example_clip_model_input,
                   compiler_workdir=VIT_CLIP_HF_COMPILATION_DIR,
                   compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={VIT_CLIP_HF_COMPILATION_DIR}/log-neuron-cc.txt'],
                   )

        neuron_model=clip_model_neuron
        file_name ="clip_neuron_hf_model.pt":
            torch_neuronx.async_load(neuron_model)
            torch_neuronx.lazy_load(neuron_model)
            torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)

    # Free up memory
    print(neuron_model.code)
    del neuron_model, example_clip_model_input

    ################# 3.2 flux model compile ##################
    FLUX_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "FLUX"
    FLUX_COMPILATION_DIR.mkdir(exist_ok=True)

    # temp for debug
    VISION_MODEL_HIDDEN_DIM = 1280
    VAE_OUT_CHANNELS = 3
    HEIGHT = 224
    WIDTH = 224

    example_flux_model_input = (img: Tensor,
                                       img_ids: Tensor,
                                       txt: Tensor,
                                       txt_ids: Tensor,
                                       timesteps: Tensor,
                                       y: Tensor,
                                       guidance: Tensor )
    with torch.no_grad():
       flux_model_neuron = torch_neuronx.trace(
            model_neuron,
            example_flux_model_input,
            compiler_workdir=FLUX_COMPILATION_DIR,
            compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={FLUX_COMPILATION_DIR}/log-neuron-cc.txt'],
            )

      neuron_model=flux_model_neuron
      file_name ="flux_neuron_model.pt":
          torch_neuronx.async_load(neuron_model)
          torch_neuronx.lazy_load(neuron_model)
          torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)
        # Free up memory
      print(neuron_model.code)
      del neuron_model, example_flux_model_input

################# 3.3 ae model compile ##################
      AE_COMPILATION_DIR = NEURON_COMPILER_WORKDIR / "AE"
      AE_COMPILATION_DIR.mkdir(exist_ok=True)

      # temp for debug
      VISION_MODEL_HIDDEN_DIM = 1280
      VAE_OUT_CHANNELS = 3
      HEIGHT = 224
      WIDTH = 224

      example_ae_model_input = torch.randn((BATCH_SIZE*NUM_IMAGES_PER_PROMPT, VAE_OUT_CHANNELS, HEIGHT, WIDTH), dtype=DTYPE)
      with torch.no_grad():
          ae_model_neuron = torch_neuronx.trace(
            ae_neuron,
            example_ae_model_input,
            compiler_workdir=AE_COMPILATION_DIR,
            compiler_args=[*NEURON_COMPILER_CLI_ARGS, f'--logfile={AE_COMPILATION_DIR}/log-neuron-cc.txt'],
            )

          neuron_model=ae_model_neuron
          file_name ="ae_neuron_model.pt":
            torch_neuronx.async_load(neuron_model)
            torch_neuronx.lazy_load(neuron_model)
            torch.jit.save(neuron_model, NEURON_COMPILER_OUTPUT_DIR / file_name)
          # Free up memory
          print(neuron_model.code)
          del neuron_model, example_flux_model_input

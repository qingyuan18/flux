import torch
import torch_xla.core.xla_model as xm
import torch
import torch_neuronx
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerCrossAttention
from transformers.generation.utils import ModelOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from optimum.neuron.generation import NeuronGenerationMixin

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from transformers.generation.utils import (
    BeamSearchOutput,
    GreedySearchOutput,
)
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerCrossAttention

class DecoderWrapper(torch.nn.Module):

    def __init__(self,
                 decoder: T5Stack,
                 lm_head: torch.nn.Linear,
                 model_config,
                 num_beams: int,
                 max_length: int,
                 device: str,
                 tp_degree=None):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim=model_config.d_model
        self.device = device
        self.num_beams = num_beams
        self.batch_size = 1
        self.config = model_config

        num_heads=model_config.num_heads
        num_decoder_layers=model_config.num_decoder_layers

        self.num_attention_heads_per_partition = num_heads

        # (num_beams, n_heads, seq_length, dim_per_head)
        if device == "cpu":
            self.past_key_values_sa = [torch.ones((num_beams,num_heads,max_length-1,model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
            self.past_key_values_ca = [torch.ones((num_beams,num_heads,max_length,model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
        elif device == "xla":
            self.past_key_values_sa = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((num_beams,self.num_attention_heads_per_partition,max_length-1,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])
            self.past_key_values_ca = torch.nn.ParameterList([torch.nn.Parameter(torch.ones((num_beams,self.num_attention_heads_per_partition,max_length,model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])

    def update_past(self, past_key_values):
        new_past_sa = []
        new_past_ca = []
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past_sa += [new_past_layer[:2],]
            new_past_ca += [new_past_layer[2:],]
        return new_past_sa, new_past_ca

    def reorder_cache(self, past_key_values, beam_idx):
        for i in range(len(past_key_values)):
            gather_index = beam_idx.view([beam_idx.shape[0],1,1,1]).expand_as(past_key_values[i])
            past_key_values[i] = torch.gather(past_key_values[i], dim = 0, index=gather_index)
        return past_key_values

    def forward(self,
                input_ids,
                decoder_attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                beam_idx,
                beam_scores,
                **kwargs):

        if self.num_beams > 1:
            # We reorder the cache based on the beams selected in each iteration. Required step for beam search.
            past_key_values_sa = self.reorder_cache(self.past_key_values_sa, beam_idx)
            past_key_values_ca = self.reorder_cache(self.past_key_values_ca, beam_idx)
        else:
            # We do not need to reorder for greedy sampling
            past_key_values_sa = self.past_key_values_sa
            past_key_values_ca = self.past_key_values_ca

        # The cache is stored in a flatten form. We order the cache per layer before passing it to the decoder.
        # Each layer has 4 tensors, so we group by 4.
        past_key_values = [[*past_key_values_sa[i*2:i*2+2], *past_key_values_ca[i*2:i*2+2]] for i in range(0, int(len(past_key_values_ca)/2))]

        decoder_output = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False)

        last_hidden_state = decoder_output['last_hidden_state']
        past_key_values = decoder_output['past_key_values']

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            last_hidden_state = last_hidden_state * (self.model_dim**-0.5)

        lm_logits = self.lm_head(last_hidden_state)

        past_key_values_sa, past_key_values_ca = self.update_past(past_key_values)

        # We flatten the cache to a single array. This is required for the input output aliasing to work
        past_key_values_sa = [vec for kv_per_layer in past_key_values_sa for vec in kv_per_layer]
        past_key_values_ca = [vec for kv_per_layer in past_key_values_ca for vec in kv_per_layer]

        if self.device == "cpu":
            self.past_key_values_sa = past_key_values_sa
            self.past_key_values_ca = past_key_values_ca

        # We calculate topk inside the wrapper
        next_token_logits = lm_logits[:, -1, :]

        if self.num_beams > 1:
            # This section of beam search is run outside the decoder in the huggingface t5 implementation.
            # To maximize the computation within the neuron device, we move this within the wrapper
            logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
            logsumexp = torch.log(torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
            next_token_scores = next_token_logits - logit_max - logsumexp
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
            next_token_scores = next_token_scores * 1

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            return [next_token_scores, next_tokens, next_indices] + past_key_values_sa + past_key_values_ca
        else:
            # Greedy
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            return [next_tokens] + past_key_values_sa + past_key_values_ca


class EncoderWrapper(torch.nn.Module):
    '''
        We will trace an instance of the EncoderWrapper.
        This wrapper just converts positional args to kwargs.
    '''

    def __init__(self,
                 encoder,
                 decoder,
                 model_config,
                 batch_size,
                 max_length,
                 device,
                 num_beams,
                 tp_degree=None):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_config = model_config
        self.device = device
        self.num_beams = num_beams
        self.num_attention_heads_per_partition = model_config.num_heads
        self.tp_degree = tp_degree

    def forward(self, input_ids, attention_mask):
        '''
            This is the core functionality we want to trace.
        '''
        encoder_output =  self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       output_attentions=False,
                                       output_hidden_states=False)

        last_hidden_state = encoder_output["last_hidden_state"]
        encoder_hidden_states = torch.concat([tensor.unsqueeze(0).repeat(self.num_beams, 1, 1) for tensor in last_hidden_state])

        decoder_blocks = self.decoder.block
        present_key_value_states_sa = []
        present_key_value_states_ca = []

        for i, block in enumerate(decoder_blocks):

            # Cross attention has to be initialized with the encoder hidden state
            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                """projection"""
                return states.view(self.batch_size, -1, self.num_attention_heads_per_partition, attention.key_value_proj_dim).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            # cross_attn_kv_state
            present_key_value_states_ca.append(key_states)
            present_key_value_states_ca.append(value_states)

            # Self attention kv states are initialized to zeros. This is done to keep the size of the kv cache tensor constant.
            # The kv cache will be an input to the decoder trace. Any traced function will have a fixed control flow. What this means
            # is that the trace performs the exact same computations on inputs of the same shape in each invocation. So the attention
            # kv cache is padded here to keep a fixed shape.
            present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     # key states
                                                            self.model_config.num_heads,
                                                            self.max_length-1,
                                                            self.model_config.d_kv), dtype=torch.float32, device=self.device))
            present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     # value states
                                                            self.model_config.num_heads,
                                                            self.max_length-1,
                                                            self.model_config.d_kv), dtype=torch.float32, device=self.device))

        return present_key_value_states_sa + present_key_value_states_ca



class T5Wrapper(T5ForConditionalGeneration, NeuronGenerationMixin):

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        encoder = self.get_encoder()
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(inputs_tensor, model_kwargs["attention_mask"])
        return model_kwargs

    # Override to cut the input_ids to just last token
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids as past is cached
        input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    '''
        We update the cache in the decoder trace, so lets override the _update_model_kwargs_for_xla_generation in NeuronGenerationMixin
    '''
    def _update_model_kwargs_for_xla_generation(
        self,
        model_kwargs: Dict[str, Any],
        batch_size: int,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        max_length: Optional[int] = None,
        seq_length: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        def _update_attention(model_kwargs, is_encoder_decoder):
            """Updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""

            attention_mask_name = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
            attention_mask = model_kwargs.pop(attention_mask_name)
            attention_mask_update_slice = torch.ones(
                (batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask[:, 1:], attention_mask_update_slice], dim=-1)
            mask = {attention_mask_name: attention_mask}
            return mask

        mask = _update_attention(model_kwargs, is_encoder_decoder)
        # sets the updated variables (mask and past_key_values)
        model_kwargs.update(mask)

        # Set a mock cache tensor
        model_kwargs["past_key_values"] = torch.tensor([])

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        '''
            This is needed for beam search and not greedy sampling
            We reorder the cache within the trace so we can skip it in modelling_t5.py. So we override the _reorder_cache
        '''
        self.beam_idx = beam_idx
        return past_key_values

    def generate(self,
                tokenizer: T5Tokenizer,
                prompt: str,
                max_length: int,
                num_beams: int,
                num_return_sequences: int,
                device: str):

        batch_encoding = tokenizer(prompt, max_length=max_length, truncation=True, padding='max_length',
                                return_tensors="pt")

        past_key_values = self.encoder(batch_encoding['input_ids'],batch_encoding['attention_mask'])

        decoder_attention_mask = torch.cat([torch.zeros((1, max_length-1), dtype=torch.int32),
                                            torch.ones((1, 1), dtype=torch.int32)], axis=1)

        # copy the new cache state to the decoder
        if device == "xla":
            for state, tensor in zip(self.decoder.parameters(), past_key_values):
                state.copy_(tensor)
        else:
            # First half of the cache is self attention and the rest is cross attention
            self.decoder.past_key_values_sa = past_key_values[:len(past_key_values)//2]
            self.decoder.past_key_values_ca = past_key_values[len(past_key_values)//2:]

        output = super().generate(**batch_encoding,
                                max_length=max_length,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                do_sample=False,
                                use_cache=True,
                                decoder_attention_mask=decoder_attention_mask,
                                encoder_outputs={"last_hidden_state": torch.ones((1,128,1))}) # Pass fake encoder_outputs so the transfomers code will not invoke the encoder
        return output

    def forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        beam_scores = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        hidden_states = encoder_outputs["last_hidden_state"]

        if not hasattr(self, 'beam_idx'):
            # Infering the number of beams from the attention mask
            num_beams = attention_mask.shape[0]
            self.beam_idx = torch.arange(0, num_beams, dtype=torch.int64)

        decoder_outputs = self.decoder(
            decoder_input_ids,
            decoder_attention_mask,
            hidden_states,
            attention_mask,
            self.beam_idx,
            beam_scores
        )

        # lm_logits = decoder_outputs[0]
        next_token_scores = decoder_outputs[0]
        next_tokens = decoder_outputs[1]
        next_indices = decoder_outputs[2]

        return next_token_scores, next_tokens, next_indices

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        seq_length: Optional[int] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # Overwrite cur_len
        cur_len = seq_length

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        # beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores_device = "cpu"
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=beam_scores_device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # prepare model inputs
            # From max_length-sized input_ids, select first
            # cur_len - 1 values.
            update_indices = torch.stack(
                [torch.arange(input_ids.size(0)), torch.tensor(cur_len - 1).repeat(input_ids.size(0))], dim=-1
            )
            input_ids_ = input_ids[update_indices[:, 0], update_indices[:, 1], None]
            model_inputs = self.prepare_inputs_for_generation(input_ids_, **model_kwargs)

            next_token_scores, next_tokens, next_indices = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                beam_scores=beam_scores
            )

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids.to("cpu")[:, :cur_len],
                next_token_scores.to("cpu"),
                next_tokens.to("cpu"),
                next_indices.to("cpu"),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            update_indices = torch.stack(
                [torch.arange(batch_beam_size), torch.tensor(cur_len - 1).repeat(batch_beam_size)], dim=-1
            )
            update_indices_2 = torch.stack(
                [torch.arange(batch_beam_size), torch.tensor(cur_len).repeat(batch_beam_size)], dim=-1
            )
            # First select beam_indices
            device = input_ids.device
            beam_idx_device = beam_idx.to(device=input_ids.device)
            input_ids[:, :] = input_ids[beam_idx_device.long(), :]

            # Then append new tokens
            input_ids[update_indices_2[:, 0], update_indices_2[:, 1], None] = beam_next_tokens.unsqueeze(-1).to(device).to(torch.long)
            input_ids = input_ids * 1  # Hack to materialize tensor

            # update generated ids, model inputs, and length for next step
            model_kwargs = self._update_model_kwargs_for_xla_generation(
                model_kwargs,
                batch_size=batch_beam_size,
                is_encoder_decoder=self.config.is_encoder_decoder,
                max_length=stopping_criteria.max_length,
                seq_length=cur_len,
                use_cache=model_kwargs["use_cache"],
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx.to(torch.int64))

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            # stop when each sentence is finished, or if we exceed the maximum length
            stop_criterion_1 = beam_scorer.is_done
            if isinstance(stopping_criteria, list):
                if len(stopping_criteria) == 1:
                    stopping_criteria = stopping_criteria[0]

            # Cases that can be handled in XLA without requiring
            # non-padded input_ids
            if isinstance(stopping_criteria, MaxLengthCriteria):
                stop_criterion_2 = cur_len >= stopping_criteria.max_length
            elif isinstance(stopping_criteria, MaxTimeCriteria):
                stop_criterion_2 = stopping_criteria(input_ids, scores)
            else:
                # Other cases will be handled on CPU
                batch_size, _ = input_ids.shape
                input_ids_cpu = input_ids.to("cpu")
                mask = torch.cat(
                    [torch.ones(batch_size, cur_len), torch.zeros(batch_size, input_ids.shape[1] - cur_len)], dim=1
                ).bool()
                input_ids_cpu = torch.masked_select(input_ids_cpu, mask).reshape((batch_size, cur_len))
                scores_cpu = scores.to("cpu") if torch.is_tensor(scores) else scores
                stop_criterion_2 = stopping_criteria(input_ids_cpu, scores_cpu)

            if stop_criterion_1 or stop_criterion_2:
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids.to("cpu"),
            beam_scores.to("cpu"),
            next_tokens.to("cpu"),
            next_indices.to("cpu"),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        for k, v in sequence_outputs.items():
            if type(v) == torch.Tensor:
                sequence_outputs[k] = sequence_outputs[k].to(input_ids.device)

        return sequence_outputs["sequences"]


    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        seq_length: Optional[int] = int,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        """
            Overriding greedy sampling to use next tokens returned from neuron device instead of logits.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        use_cache = model_kwargs["use_cache"] if "use_cache" in model_kwargs else False
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None


        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:

            # prepare model inputs
            # From max_length-sized input_ids, select first
            # seq_length - 1 values.

            if model_kwargs.get("past_key_values") is None:
                input_ids_ = input_ids[:, :seq_length]
            else:
                update_indices = torch.stack(
                    [torch.arange(input_ids.size(0)), torch.tensor(seq_length - 1).repeat(input_ids.size(0))],
                    dim=-1,
                )
                input_ids_ = input_ids[update_indices[:, 0], update_indices[:, 1], None]

            model_inputs = self.prepare_inputs_for_generation(input_ids_, **model_kwargs)

            # forward pass to get next token
            output = self(
               **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_tokens = output[0]

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step

            batch_size, _ = input_ids.shape
            update_indices = torch.stack(
                [torch.arange(batch_size), torch.tensor(seq_length).repeat(batch_size)], dim=-1
            )
            input_ids[update_indices[:, 0], update_indices[:, 1]] = next_tokens[:]
            model_kwargs = self._update_model_kwargs_for_xla_generation(
                model_kwargs,
                batch_size=batch_size,
                is_encoder_decoder=self.config.is_encoder_decoder,
                max_length=stopping_criteria.max_length,
                seq_length=seq_length,
                use_cache=use_cache,
            )

            seq_length += 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            stop_criterion_1 = unfinished_sequences.max() == 0

            if isinstance(stopping_criteria, list):
                if len(stopping_criteria) == 1:
                    stopping_criteria = stopping_criteria[0]

            # Cases that can be handled in XLA without requiring
            # non-padded input_ids
            if isinstance(stopping_criteria, MaxLengthCriteria):
                stop_criterion_2 = seq_length >= stopping_criteria.max_length
            elif isinstance(stopping_criteria, MaxTimeCriteria):
                stop_criterion_2 = stopping_criteria(input_ids, scores)
            else:
                # Other cases will be handled on CPU
                batch_size, _ = input_ids.shape
                mask = torch.cat(
                    [torch.ones(batch_size, seq_length), torch.zeros(batch_size, input_ids.shape[1] - seq_length)],
                    dim=1,
                ).bool()
                input_ids_cpu = torch.masked_select(input_ids, mask).reshape((batch_size, seq_length)).to("cpu")
                scores_cpu = scores.to("cpu") if torch.is_tensor(scores) else scores
                stop_criterion_2 = stopping_criteria(input_ids_cpu, scores_cpu)

            if stop_criterion_1 or stop_criterion_2:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        return input_ids


def trace_encoder(model: T5ForConditionalGeneration,
                  tokenizer: T5Tokenizer,
                  max_length: int,
                  num_beams: int):

    # Trace encoder
    batch_encoding = tokenizer("translate English to German: Lets go home now",
                               max_length=max_length, truncation=True, padding='max_length', return_tensors="pt")
    input_ids = batch_encoding['input_ids']
    attention_mask = batch_encoding['attention_mask']

    encoder = EncoderWrapper(model.encoder, model.decoder, model.config, num_beams, max_length, "xla", num_beams)
    traced_encoder = torch_neuronx.trace(encoder, (input_ids, attention_mask), compiler_workdir="/tmp/encoder/")
    setattr(traced_encoder, 'main_input_name', 'input_ids')  # Attribute required by beam search

    return traced_encoder

def trace_decoder(model: T5ForConditionalGeneration,
                  num_beams: int,
                  max_length: int):

    decoder = DecoderWrapper(decoder=model.decoder,
                             lm_head=model.lm_head,
                             model_config=model.config,
                             num_beams=num_beams,
                             max_length=max_length,
                             device="xla")

    # We create mock inputs so we can trace the decoder
    decoder_input_ids = torch.ones((num_beams, 1), dtype=torch.int64)
    decoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int32)
    encoder_attention_mask = torch.ones((num_beams, max_length), dtype=torch.int64)
    encoder_hidden_states = torch.ones((num_beams, max_length, model.config.d_model), dtype=torch.float32)

    beam_idx = torch.arange(0, num_beams, dtype=torch.int64)
    beam_scores = torch.zeros((num_beams,), dtype=torch.float)

    num_outputs_from_trace = 3 if num_beams > 1 else 1

    aliases = {}
    for i in range(len(decoder.past_key_values_sa)):
        aliases[decoder.past_key_values_sa[i]] = i + num_outputs_from_trace
    for i in range(len(decoder.past_key_values_ca)):
        aliases[decoder.past_key_values_ca[i]] = len(decoder.past_key_values_sa) + i + num_outputs_from_trace

    traced_decoder = torch_neuronx.trace(decoder, (
        decoder_input_ids,
        decoder_attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        beam_idx,
        beam_scores,
    ), input_output_aliases=aliases, compiler_workdir="/tmp/decoder/")

    return traced_decoder

def main():
    # Let's set some run parameters
    NEURON_COMPILER_WORKDIR = Path("neuron_compiler_workdir")
    NEURON_COMPILER_WORKDIR.mkdir(exist_ok=True)
    NEURON_COMPILER_OUTPUT_DIR = Path("compiled_models")
    NEURON_COMPILER_OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # We enable this flag to ensure model uses attention key value caching
    model.config.use_cache = True
    traced_encoder = trace_encoder(model, tokenizer, max_length, num_beams)
    traced_decoder = trace_decoder(model, num_beams, max_length)

    torch.jit.save(traced_encoder, NEURON_COMPILER_OUTPUT_DIR / "TracedEncoder.pt")
    torch.jit.save(traced_decoder, NEURON_COMPILER_OUTPUT_DIR / "TracedDecoder.pt")


    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()
    runtime.set_default_neuron_cores(0, 1)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Wrapper.from_pretrained(model_name)

    model.encoder = torch.jit.load(NEURON_COMPILER_OUTPUT_DIR / "TracedEncoder.pt")
    # Attribute required by beam search
    setattr(model.encoder, 'main_input_name', 'input_ids')

    model.decoder = torch.jit.load(NEURON_COMPILER_OUTPUT_DIR / "TracedDecoder.pt")
    torch_neuronx.move_trace_to_device(model.decoder, 0)


    ### inference

    #output = model.generate(tokenizer=tokenizer,
    #                        prompt="translate English to German: Lets eat good food.",
    #                        max_length=max_length,
    #                        num_beams=num_beams,
    #                        num_return_sequences=num_return_sequences,
    #                        device="xla")
    #
    #results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    #
    #print('Results:')
    #for i, summary in enumerate(results):
    #    print(i + 1, summary)

if __name__ == "__main__":
   main()
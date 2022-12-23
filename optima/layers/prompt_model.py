import torch
import torch.nn as nn
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from transformers.modeling_utils import PreTrainedModel
from openprompt.prompt_base import Template, Verbalizer
from openprompt.pipeline_base import PromptForClassification
from openprompt.utils import round_list, signature
from mymodules.utils import accuracy_fct
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from torch.nn import CrossEntropyLoss
import warnings


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
class MyT5Model(T5ForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        hidden_states = torch.cat([self.soft_embeds, hidden_states], 1)
        attention_mask = torch.cat([torch.ones(self.soft_embeds.size()[0:2], dtype=attention_mask.dtype, device=attention_mask.device),
                                    attention_mask], dim=-1)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



class MyPromptModelForClassification(PromptForClassification):
    def __init__(self,
                 domain_disc: None,
                 attention: None,
                 plm: PreTrainedModel,
                 verbalizer: Verbalizer,
                 template: Template,
                 prompt_layer: int=0,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False
                 ):
        super().__init__(
            plm=plm,
            verbalizer=verbalizer,
            template=template,
            freeze_plm=freeze_plm,
            plm_eval_mode=plm_eval_mode
        )
        self.plm = plm
        self.plm.soft_embeds.requires_grad_(True)  # TODO

        self.verbalizer = verbalizer
        self.template = template
        self.prompt_layer = prompt_layer
        self.plm_eval_mode = plm_eval_mode
        self.freeze_plm = freeze_plm
        self.forward_keys = signature(self.plm.forward).args

        self.domain_disc = domain_disc
        self.attention = attention
        self.loss_fct = nn.CrossEntropyLoss()
        self.accuracy_fct = accuracy_fct

    def input_ids_to_embeds(self, batch):
        # TODO input_ids dropped for T5 forward: https://huggingface.co/transformers/v3.1.0/_modules/transformers/modeling_t5.html#T5Model
        batch = self.template.input_ids_to_embeds(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

    def forward_computation(self, input_batch, loss_ids, label):
        outputs = self.plm(**input_batch, output_hidden_states=True)

        outputs = self.template.post_processing_outputs(outputs)  # plm hidden states

        outputs_at_mask = outputs.logits[torch.where(loss_ids > 0)]  # logits at lm_head
        label_words_logits = self.verbalizer.process_logits(outputs_at_mask)  # return B * num_classes matrix

        loss = self.loss_fct(label_words_logits, label)
        return loss, label_words_logits, outputs

    def forward(self, batch: Union[Dict, InputFeatures]):
        r"""
        Get the logits of label words.
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): ['inputs_embeds', 'attention_mask', 'label', 'decoder_input_ids', 'loss_ids', 'guid']
            batch.decoder_input_ids: [0, 32099, 1] -> pad, sentinel(extra_id_0), </s>
            batch.loss_ids: [0, 1, 0]
        Returns:
            :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
        """
        input_batch = self.input_ids_to_embeds(batch)

        return self.forward_computation(input_batch, batch["loss_ids"], batch["label"])


    def state_dict(self, *args, **kwargs):
        """ Save the model using template and plm's save methods. """
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)

        _state_dict['template'] = self.template.state_dict(*args, **kwargs)
        if self.prompt_model.freeze_plm:
            _state_dict['template'].pop("raw_embedding.weight")

        return _state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """ Load the model using template and plm's load methods. """
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)

        state_dict['template']["raw_embedding.weight"] = self.plm.encoder.embed_tokens.weight
        self.template.load_state_dict(state_dict['template'], *args, **kwargs)


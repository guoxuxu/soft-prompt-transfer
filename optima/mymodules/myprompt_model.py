import torch
import torch.nn as nn
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.prompt_base import Template, Verbalizer
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils import round_list, signature
from mymodules.utils import accuracy_fct


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None


def grad_reverse(x, constant):
    return GradReverse.apply(x, constant)


def save_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


class MyPromptModelForClassification(PromptForClassification):
    def __init__(self,
                 domain_disc: None,
                 plm: PreTrainedModel,
                 verbalizer: Verbalizer,
                 template: Template,
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
        self.verbalizer = verbalizer
        self.template = template
        self.plm_eval_mode = plm_eval_mode
        self.freeze_plm = freeze_plm
        self.forward_keys = signature(self.plm.forward).args

        self.domain_disc = domain_disc
        self.loss_fct = nn.CrossEntropyLoss()
        self.accuracy_fct = accuracy_fct

    def input_ids_to_embeds(self, batch):
        # TODO input_ids dropped for T5 forward: https://huggingface.co/transformers/v3.1.0/_modules/transformers/modeling_t5.html#T5Model
        batch = self.template.input_ids_to_embeds(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

    def concat_prompts(self, input_batch):
        input_batch = self.template.process_batch(input_batch)
        return input_batch

    def forward_computation(self, input_batch, loss_ids, label):
        input_batch = self.concat_prompts(input_batch)
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)  # plm hidden states

        outputs_at_mask = outputs.logits[torch.where(loss_ids > 0)]  # logits at lm_head
        label_words_logits = self.verbalizer.process_logits(outputs_at_mask)  # return B * num_classes matrix, see self.verbalizer.label_words_ids

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
        # # logits = self.verbalizer.gather_outputs(outputs)  # return outputs.logits, this one does not generate next token (model.generate)

        return self.forward_computation(input_batch, batch["loss_ids"], batch["label"])


    def domain_discrimination(self, outputs, loss_ids, device, domain: str, ad_weight: float):
        batch_size = outputs.logits.size(0)
        num_layers = len(outputs.encoder_hidden_states)  # 13

        encoder_hidden_states = outputs.encoder_hidden_states
        decoder_hidden_states = ()
        if self.template.model_is_encoder_decoder:
            for i in range(0, len(outputs.decoder_hidden_states)):
                decoder_hidden_states += (outputs.decoder_hidden_states[i][torch.where(loss_ids > 0)],)

        if domain == "src":
            disc_labels = torch.ones(batch_size, dtype=torch.long)
        elif domain == "tgt":
            disc_labels = torch.zeros(batch_size, dtype=torch.long)
        else:
            raise NotImplementedError

        hidden_states = (decoder_hidden_states[-1], )  # dec bert last transformer layer
        if type(hidden_states) is tuple:
            hidden_states = (h.to(device) for h in hidden_states)
        else:
            raise NotImplementedError

        tot_disc_loss = 0
        tot_disc_acc = 0
        for i, (disc_layer, layer_hidden) in enumerate(zip(self.domain_disc, hidden_states)):
            disc_logits = disc_layer(grad_reverse(layer_hidden, ad_weight))
            disc_loss = self.loss_fct(disc_logits, disc_labels.to(disc_logits.device))
            tot_disc_loss += disc_loss
            tot_disc_acc += self.accuracy_fct(disc_logits, disc_labels)

        domain_disc_loss = tot_disc_loss / len(self.domain_disc)
        domain_disc_acc = tot_disc_acc / len(self.domain_disc)
        return domain_disc_loss, domain_disc_acc


    def state_dict(self, *args, **kwargs):
        """ Save the model using template and plm's save methods. """
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict(*args, **kwargs)

        _state_dict['template'] = self.template.state_dict(*args, **kwargs)
        if self.freeze_plm:
            _state_dict['template'].pop("raw_embedding.weight")
        if "domain_disc" in [n for n, p in self.named_children()]:
            _state_dict['domain_disc'] = self.domain_disc.state_dict(*args, **kwargs)

        return _state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """ Load the model using template and plm's load methods. """
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'], *args, **kwargs)

        state_dict['template']["raw_embedding.weight"] = self.plm.encoder.embed_tokens.weight
        self.template.load_state_dict(state_dict['template'], *args, **kwargs)

        if 'domain_disc' in state_dict:
            self.domain_disc.load_state_dict(state_dict['domain_disc'], *args, **kwargs)




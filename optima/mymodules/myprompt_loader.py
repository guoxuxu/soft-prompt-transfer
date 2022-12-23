
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from openprompt.utils import round_list, signature
from torch.utils.data import DataLoader
from collections import defaultdict


class MyPromptDataset(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class: TokenizerWrapper,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[int] = 512,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 **kwargs,
                 ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length" : max_seq_length,
            "truncate_method" : truncate_method,
            "decoder_max_length" : decoder_max_length,
            "predict_eos_token" : predict_eos_token,
            "tokenizer" : tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
                # text = "".join([wrapped_example[0][i]["text"] for i in range(len(wrapped_example[0])) if wrapped_example[0][i]["text"] != self.tokenizer_wrapper.template_mask_token])
                # text_len = len(self.tokenizer_wrapper.tokenizer.encode(text, add_special_tokens=False))
                # if text_len >= 10 and text_len <= self.tokenizer_wrapper.max_seq_length:
                #     self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
           T5 does not make use of token type ids, therefore a list of zeros is returned: https://huggingface.co/docs/transformers/model_doc/t5
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset), desc='tokenizing'):
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            # inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            inputfeatures = InputFeatures(**self.tokenize_one_example(wrapped_example), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def process(self):
        # processs
        self.wrap()
        self.tokenize()
        return self.tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def tokenize_one_example(self, wrapped_example):
        ''' # TODO doens't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0

        decoder_input_ids = []
        loss_ids = []

        # text = "".join([wrapped_example[0][i]["text"] for i in range(len(wrapped_example[0])) if wrapped_example[0][i]["text"] != self.tokenizer_wrapper.template_mask_token])
        # encoder_inputs['input_ids'] = self.tokenizer_wrapper.tokenizer.encode(text, add_special_tokens=False)

        for piece_id, piece in enumerate(wrapped_example):
            if piece['text'] == self.tokenizer_wrapper.template_mask_token:

                decoder_input_ids.append(self.tokenizer_wrapper.mask_token_ids(num_mask_token_used))
                encode_text = [self.tokenizer_wrapper.mask_token_ids(num_mask_token_used)]
                # decoder_input_ids.append(self.mask_token_ids(num_mask_token_used+1))
                # loss_ids[-1] = 1 # shift loss_ids
                loss_ids.append(1)
                num_mask_token_used += 1
            else:
                encode_text = self.tokenizer_wrapper.tokenizer.encode(piece['text'], add_special_tokens=False)

            encoding_length = len(encode_text)

            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text', 'loss_ids']:
                    encoder_inputs[key].append([piece[key]] * encoding_length)


        # decoder input ids
        decoder_inputs = {'decoder_input_ids': decoder_input_ids, 'loss_ids': loss_ids}
        decoder_inputs = self.tokenizer_wrapper.truncate_decoder_inputs(decoder_inputs)

        encoder_inputs = self.tokenizer_wrapper.truncate(encoder_inputs=encoder_inputs)
        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.tokenizer_wrapper.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.tokenizer_wrapper.add_special_tokens(encoder_inputs=encoder_inputs)

        # create special input ids
        encoder_inputs['attention_mask'] = [1] * len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.tokenizer_wrapper.padding(input_dict=encoder_inputs,
                                                        max_len=self.tokenizer_wrapper.max_seq_length,
                                                        pad_id_for_inputs=self.tokenizer_wrapper.tokenizer.pad_token_id)

        all_input_ids = {**encoder_inputs, **decoder_inputs}
        return all_input_ids


class MyPromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 tokenizer: PreTrainedTokenizer,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 drop_last: Optional[bool] = False,
                 ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.tensor_dataset = dataset
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.batch_size = batch_size

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self ,):
        return self.dataloader.__iter__()
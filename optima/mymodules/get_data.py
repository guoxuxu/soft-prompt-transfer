from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
import os, torch


class SICKFewShotProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                idx = linelist[1]
                text_a = linelist[2]
                text_b = linelist[3]
                label = linelist[4]
                examples.append({"sentence_A": text_a, "sentence_B": text_b, "label": label, "idx": idx})
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['sentence_A']
        text_b = example['sentence_B']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class SICKProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split == "dev":
            split = "test"  # TODO changed validation to test
        dataset = torch.load(os.path.join(data_dir, "sick.pt"))[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['sentence_A']
        text_b = example['sentence_B']
        label = int(example['label'])
        guid = "{}".format(example['id'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class MultiRCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split == "dev":
            split = "validation"
        dataset = torch.load(os.path.join(data_dir, "multirc.pt"))[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['paragraph']
        text_b = example['question']
        meta = {"answer": example['answer']}
        label = example['label']
        idx = f"{example['idx']['paragraph']}_{example['idx']['question']}_{example['idx']['answer']}"
        guid = "{}".format(idx)
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label)


class MultiRCFewShotProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split == "test":
            dataset = torch.load(os.path.join("/".join(data_dir.split("/")[0:-1]), "test.pt"))
        else:
            dataset = torch.load(os.path.join(data_dir, f"{split}.pt"))
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['paragraph']
        text_b = example['question']
        meta = {"answer": example['answer']}
        label = example['label']
        idx = f"{example['idx']['paragraph']}_{example['idx']['question']}_{example['idx']['answer']}"
        guid = "{}".format(idx)
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label)



class CBProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split == "dev":
            split = "validation"
        dataset = torch.load(os.path.join(data_dir, "cb.pt"))[split]
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class CBFewShotProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if split == "test":
            dataset = torch.load(os.path.join("/".join(data_dir.split("/")[0:-1]), "test.pt"))
        else:
            dataset = torch.load(os.path.join(data_dir, f"{split}.pt"))
        return list(map(self.transform, dataset))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)



class QQPProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')  # QQP
                if len(linelist) > 6:
                    raise ValueError
                label = linelist[-1]
                text_a = linelist[3]
                text_b = linelist[4]
                examples.append({"question1": text_a, "question2": text_b, "label":label, "idx":idx})
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['question1']
        text_b = example['question2']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class MRPCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        if "msr_paraphrase" in split:
            path = os.path.join(data_dir, f"{split}.txt")
        else:
            path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')  # MRPC
                if len(linelist) > 5:
                    raise ValueError
                label = linelist[0]
                text_a = linelist[3]
                text_b = linelist[4]
                examples.append({"sentence1": text_a, "sentence2": text_b, "label":label, "idx":idx})
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['sentence1']
        text_b = example['sentence2']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class MRPCFewShotProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')  # MRPC
                if len(linelist) > 5:
                    raise ValueError
                label = linelist[0]
                text_a = linelist[3]
                text_b = linelist[4]
                examples.append({"sentence1": text_a, "sentence2": text_b, "label":label, "idx":idx})
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['sentence1']
        text_b = example['sentence2']
        label = int(example['label'])
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)



"""MNLI has different genres, the column is named genre. Category and cross-genre"""


class MNLIProcessor(DataProcessor):  # MNLI
    def __init__(self):
        super().__init__()
        self.labels = ['contradiction', 'neutral', 'entailment']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')    # MNLI
                try:
                    label = self.labels.index(linelist[-1])
                except:
                    print(linelist)
                    continue
                text_a = linelist[8]
                text_b = linelist[9]
                examples.append({"premise": text_a, "hypothesis": text_b, "label":label, "idx":idx})
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = example['label']
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


class SNLIProcessor(DataProcessor):   # SNLI
    def __init__(self):
        super().__init__()
        self.labels = ['contradiction', 'neutral', 'entailment']

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')  # SNLI
                try:
                    label = self.labels.index(linelist[-1])
                except:
                    print(linelist)
                    continue
                text_a = linelist[7]
                text_b = linelist[8]
                examples.append({"premise": text_a, "hypothesis": text_b, "label":label, "idx":idx})  # TODO check sentence1 is premise
        return list(map(self.transform, examples))

    def transform(self, example):
        text_a = example['premise']
        text_b = example['hypothesis']
        label = example['label']
        guid = "{}".format(example['idx'])
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


def get_data_setting(path, dataset_name):
    dataset = {}
    if dataset_name == "snli":  # snli has dev, test
        Processor = SNLIProcessor
        data_dir = os.path.join(path, "SNLI")

    elif dataset_name in ["mnli_matched", "mnli_mismatched"]:  # mnli only has dev. mnli->snli
        Processor = MNLIProcessor
        data_dir = os.path.join(path, "MNLI")
        if dataset_name == "sst2":
            data_dir = os.path.join(path, "SST-2")
        elif dataset_name == "gsst2":
            data_dir = os.path.join(path, "GLUE-SST-2")
        else:
            raise TypeError

    elif dataset_name == "mrpc" or dataset_name == "mrpc_pp":
        Processor = MRPCProcessor
        data_dir = os.path.join(path, "MRPC")  # mrpc has dev, mrpc pp has test
    elif dataset_name == "qqp":
        Processor = QQPProcessor
        data_dir = os.path.join(path, "QQP")  # qqp only has dev. qqp->mrpc
    elif dataset_name == "sick":
        Processor = SICKProcessor
        data_dir = os.path.join(path, "SICK")
    elif dataset_name == "multirc":
        Processor = MultiRCProcessor
        data_dir = os.path.join(path, "MultiRC")
    elif dataset_name == "cb":
        Processor = CBProcessor
        data_dir = os.path.join(path, "CB")
    else:
        raise NotImplementedError

    if dataset_name == "mnli_mismacthed" or dataset_name == "mnli_matched":
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev_matched")  # only matched available
        dataset['test'] = []

    elif dataset_name == "mr":
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="test")  # mr doesn't have dev set
        dataset['test'] = []

    elif dataset_name == "mrpc_pp":
        dataset['train'] = Processor().get_examples(data_dir, split="msr_paraphrase_train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev")
        dataset['test'] = Processor().get_examples(data_dir, split="msr_paraphrase_test")

    elif dataset_name == "imdb":
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="test")
        dataset['test'] = []

    else:
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev")
        dataset['test'] = []

    return dataset


def get_few_shot_data_setting(path, dataset_name, short_dir=""):
    dataset = {}
    if dataset_name == "snli":  # snli has dev, test
        Processor = SNLIProcessor
        data_dir = os.path.join(path, "SNLI", short_dir)
    elif dataset_name in ["mnli_matched", "mnli_mismatched"]:  # mnli only has dev. mnli->snli
        Processor = MNLIProcessor
        data_dir = os.path.join(path, "MNLI", short_dir)
    elif dataset_name == "mrpc":
        Processor = MRPCFewShotProcessor
        data_dir = os.path.join(path, "MRPC", short_dir)  # mrpc has dev
    elif dataset_name == "mrpc_pp":
        Processor = MRPCFewShotProcessor
        data_dir = os.path.join(path, "MRPC_pp", short_dir)  # mrpcpp has dev, test
    elif dataset_name == "qqp":
        Processor = QQPProcessor
        data_dir = os.path.join(path, "QQP", short_dir)  # qqp only has dev. qqp->mrpc
    elif dataset_name == "cb":
        Processor = CBFewShotProcessor
        data_dir = os.path.join(path, "CB", short_dir)
    elif dataset_name == "sick":
        Processor = SICKFewShotProcessor
        data_dir = os.path.join(path, "SICK", short_dir)
    elif dataset_name == "multirc":
        Processor = MultiRCFewShotProcessor
        data_dir = os.path.join(path, "MultiRC", short_dir)
    else:
        raise NotImplementedError

    if dataset_name == "mnli_matched":
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev_matched")  # only matched available
        dataset['test'] = Processor().get_examples(data_dir, split="test_matched")

    elif dataset_name == "mnli_mismacthed":
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev_matched")  # only matched available
        dataset['test'] = Processor().get_examples(data_dir, split="test_mismatched")
    else:
        dataset['train'] = Processor().get_examples(data_dir, split="train")
        dataset['validation'] = Processor().get_examples(data_dir, split="dev")
        dataset['test'] = Processor().get_examples(data_dir, split="test")

    return dataset


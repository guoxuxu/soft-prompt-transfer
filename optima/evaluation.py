import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from collections import Counter


def evaluate(prompt_model, writer, device, args, dataloader, data_set, domain, log=False, global_step=0):
    predictions = []
    ground_truths = []
    tot_loss = 0
    tot_domain_disc_loss = 0
    prompt_model.eval()

    ebar = tqdm(total=len(dataloader), desc=f"Test: ")
    with torch.no_grad():

        for step, inputs in enumerate(dataloader):
            for k, v in inputs.items():
                if type(v) is torch.Tensor:
                    inputs[k] = v.to(device)
            loss, logits, outputs = prompt_model(inputs)
            labels = inputs['label']
            preds = torch.argmax(logits, dim=-1)
            tot_loss += loss.item() * len(labels)
            ground_truths.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

            ebar.set_postfix({'prediction': predictions[0], 'ground_truth': ground_truths[0]})
            ebar.update(1)

            if log:
                for i in range(len(labels)):
                    text = dataloader.tokenizer.decode(dataloader.tensor_dataset[int(inputs.guid[i])]['input_ids'].tolist())
                    text = text[0: text.find("</s>")]
                    writer.add_scalar(tag=f"{domain}/{data_set}/{labels[i].item()}: {text}", scalar_value=preds[i], global_step=global_step)
            if args.ad_ramp:
                domain_disc_loss, domain_disc_acc = prompt_model.domain_discrimination(outputs=outputs, loss_ids=inputs["loss_ids"],
                                                                      device=loss.device, domain=domain, ad_weight=1.0)
                tot_domain_disc_loss += domain_disc_loss.item() * len(labels)

    metrics = classification_report(y_true=ground_truths, y_pred=predictions, output_dict=True, zero_division=1)
    # acc = sum([int(i == j) for i, j in zip(predictions, ground_truths)]) / len(predictions)
    ave_loss = tot_loss / len(dataloader.tensor_dataset)

    ave_domain_disc_loss = tot_domain_disc_loss / len(dataloader.tensor_dataset) if args.ad_ramp else 0

    ACC = metrics["accuracy"]
    ACC = round(ACC * 100, 2)
    if args.tgt_data in ["snli", "mnli_matched", "mnli_mismatched", "sick", "paws"]:
        F1 = metrics["macro avg"]["f1-score"]
        if args.test:
            print(f"class 0 >>> f1: {metrics['0']['f1-score']:.4f}, precision: {metrics['0']['precision']:.4f}, recall:{metrics['0']['recall']:.4f}, support:{metrics['0']['support']}")
            print(f"class 1 >>> f1: {metrics['1']['f1-score']:.4f}, precision: {metrics['1']['precision']:.4f}, recall:{metrics['1']['recall']:.4f}, support:{metrics['1']['support']}")
            print(f"class 2 >>> f1: {metrics['2']['f1-score']:.4f}, precision: {metrics['2']['precision']:.4f}, recall:{metrics['2']['recall']:.4f}, support:{metrics['2']['support']}")
    else:
        F1 = metrics["1"]["f1-score"]
    F1 = round(F1 * 100, 2)
    return ACC, F1, ave_loss, ave_domain_disc_loss


def qqp_evaluate(prompt_model, device, args, dataloader):
    predictions = []
    ground_truths = []
    tot_loss = 0
    prompt_model.eval()

    ebar = tqdm(total=len(dataloader), desc=f"Test: ")
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            for k, v in inputs.items():
                if type(v) is torch.Tensor:
                    inputs[k] = v.to(device)
            loss, logits, outputs = prompt_model(inputs)
            labels = inputs['label']
            preds = torch.argmax(logits, dim=-1)
            tot_loss += loss.item() * len(labels)
            ground_truths.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

            ebar.set_postfix({'prediction': predictions[0], 'ground_truth': ground_truths[0], 'time': format_time()})
            ebar.update(1)

    metrics = classification_report(y_true=ground_truths, y_pred=predictions, output_dict=True, zero_division=1)

    ACC = metrics["accuracy"]
    ACC = round(ACC * 100, 2)

    assert args.tgt_data in ["mrpc", "qqp"]
    F1 = metrics["macro avg"]["f1-score"]

    f1_results = {0: metrics['0']['f1-score'], 1: metrics['1']['f1-score']}
    precision_results = {0: metrics['0']['precision'], 1: metrics['1']['precision']}
    recall_results = {0: metrics['0']['recall'], 1: metrics['1']['recall']}
    supports = {0: metrics['0']['support'], 1: metrics['1']['support']}
    all_preds = {
        0: {k: (v / supports[0]) for k, v in Counter(np.array(predictions)[np.where(np.array(ground_truths) == 0)[0]]).items()},
        1: {k: (v / supports[1]) for k, v in Counter(np.array(predictions)[np.where(np.array(ground_truths) == 1)[0]]).items()},
    }
    # F1 = round(F1 * 100, 2)
    return ACC, f1_results, precision_results, recall_results, supports, all_preds


def nli_evaluate(prompt_model, device, args, dataloader):
    predictions = []
    ground_truths = []
    tot_loss = 0
    prompt_model.eval()

    ebar = tqdm(total=len(dataloader), desc=f"Test: ")
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            for k, v in inputs.items():
                if type(v) is torch.Tensor:
                    inputs[k] = v.to(device)
            loss, logits, outputs = prompt_model(inputs)
            labels = inputs['label']
            preds = torch.argmax(logits, dim=-1)
            tot_loss += loss.item() * len(labels)
            ground_truths.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())

            ebar.set_postfix({'prediction': predictions[0], 'ground_truth': ground_truths[0], 'time': format_time()})
            ebar.update(1)

    metrics = classification_report(y_true=ground_truths, y_pred=predictions, output_dict=True, zero_division=1)

    ACC = metrics["accuracy"]
    ACC = round(ACC * 100, 2)

    assert args.tgt_data in ["snli", "mnli_matched", "mnli_mismatched", "sick", "paws", "cb"]
    F1 = metrics["macro avg"]["f1-score"]

    f1_results = {0:metrics['0']['f1-score'], 1:metrics['1']['f1-score'], 2:metrics['2']['f1-score']}
    precision_results = {0:metrics['0']['precision'], 1:metrics['1']['precision'], 2:metrics['2']['precision']}
    recall_results = {0:metrics['0']['recall'], 1:metrics['1']['recall'], 2:metrics['2']['recall']}
    supports = {0:metrics['0']['support'], 1:metrics['1']['support'], 2:metrics['2']['support']}

    all_preds = {
        0:{0:0, 1:0, 2:0},
        1:{0:0, 1:0, 2:0},
        2:{0:0, 1:0, 2:0}
    }
    all_preds[0].update({k:(v/supports[0]) for k,v in Counter(np.array(predictions)[np.where(np.array(ground_truths)==0)[0]]).items()})
    all_preds[1].update({k:(v/supports[1]) for k,v in Counter(np.array(predictions)[np.where(np.array(ground_truths)==1)[0]]).items()})
    all_preds[2].update({k:(v/supports[2]) for k,v in Counter(np.array(predictions)[np.where(np.array(ground_truths)==2)[0]]).items()})

    # F1 = round(F1 * 100, 2)
    return ACC, f1_results, precision_results, recall_results, supports, all_preds



def predict(prompt_model, device, dataloader):
    indices = []
    predictions = []
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            for k, v in inputs.items():
                if type(v) is torch.Tensor:
                    inputs[k] = v.to(device)
            logits, outputs = prompt_model(inputs)
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().tolist())
            indices.extend([int(i) for i in inputs["guid"]])

    return indices, predictions

from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate



def qa_evaluate(prompt_model, device, dataloader, decoder_max_length, min_length):
    generation_arguments = {
        "max_length": decoder_max_length,  # default: model.config.max_length
        "max_new_tokens": None,  # default value. The maximum numbers of tokens to generate, ignore the current number of tokens.
        "min_length": min_length,  # default 10
        "num_beams": 5,  # default 1
        "top_p": 0.9,  # default 1.0
        "top_k": 0,  # it's default value
        "temperature": 1.0,  # it's default value
        "do_sample": False,  # it's default value
        "repetition_penalty": 1.0,  # it's default value
    }
    predictions = []
    ground_truths = []
    prompt_model.eval()
    with torch.no_grad():
        ebar = tqdm(total=len(dataloader), desc=f"Test: ")

        for step, inputs in enumerate(dataloader):
            # inputs = inputs.to(device)
            for k, v in inputs.items():
                if type(v) is torch.Tensor:
                    inputs[k] = v.to(device)
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments, verbose=False)
            predictions.extend(output_sentence)
            ground_truths.extend(inputs['tgt_text'])

            ebar.set_postfix({'prediction': predictions[0], 'ground_truth': ground_truths[0], 'time': format_time()})
            ebar.update(1)

            # for i in range(len(inputs)):
            #     print(f"{inputs['guid'][i]}\t{prompt_model.tokenizer.decode(inputs['input_ids'][i])[0:100]}\t{inputs['tgt_text'][i]}\t{output_sentence[i]}")

    assert len(predictions) == len(ground_truths), (len(predictions), len(ground_truths))
    predictions = [prediction.strip() for prediction in predictions]
    ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    # shown one example
    F1 = crossfit_evaluate(predictions, ground_truths, metric="QA-F1")
    EM = crossfit_evaluate(predictions, ground_truths, metric="EM")
    print(f"len:{len(dataloader)}, F1:{F1}, EM:{EM},\tpredictions {predictions[0]}, ground_truths {ground_truths[0]}")

    return F1, EM


def T5_large_eval(device, dataloader):
    from transformers import AutoModelWithLMHead, AutoTokenizer
    model = AutoModelWithLMHead.from_pretrained("t5-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    predictions, ground_truths = [], []
    ebar = tqdm(total=len(dataloader), desc=f"Test: ")
    for step, inputs in enumerate(dataloader):
        for k, v in inputs.items():
            if type(v) is torch.Tensor:
                inputs[k] = v.to(device)

        input_texts = []
        for x in inputs.input_ids:
            input_texts.append(tokenizer.decode(x))
        features = tokenizer(input_texts, return_tensors='pt', truncation=True, padding=True)
        out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
        output_sentence = []
        for x in out:
            output_sentence.append(tokenizer.decode(x))

        predictions.extend(output_sentence)
        ground_truths.extend(inputs['tgt_text'])

        ebar.set_postfix({'prediction': predictions[0], 'ground_truth': ground_truths[0], 'time': format_time()})
        ebar.update(1)

    assert len(predictions) == len(ground_truths), (len(predictions), len(ground_truths))
    predictions = [prediction.strip() for prediction in predictions]
    ground_truths = [ground_truth.strip() for ground_truth in ground_truths]
    # shown one example
    F1 = crossfit_evaluate(predictions, ground_truths, metric="QA-F1")
    EM = crossfit_evaluate(predictions, ground_truths, metric="EM")
    print(f"len:{len(dataloader)}, F1:{F1}, EM:{EM},\tpredictions {predictions[0]}, ground_truths {ground_truths[0]}")

    return F1, EM
import os, torch
from mymodules.get_data import get_data_setting, get_few_shot_data_setting
from mymodules.myprompt_loader import MyPromptDataset
from mymodules.mysoft_template import MySoftTemplate
from openprompt.prompts.manual_verbalizer import ManualVerbalizer


def get_embeds(init_vocab, tokens_from, src_soft_num, tgt_soft_num, tokenizer, raw_embedding):
    if init_vocab:
        vocab = tokenizer.get_vocab()
        if tokens_from == "first":
            src_embeds = raw_embedding.weight[0: src_soft_num].clone()
            tgt_embeds = raw_embedding.weight[0: tgt_soft_num].clone()
            # print(f"Shared tokens:\n{islice(vocab.keys(), args.sha_num)}\n Unshared tokens:\n{islice(vocab.keys(), args.sha_num + args.src_soft_num)}\n")
        elif tokens_from == "firstWords":
            token_list, token_index_list = [], []
            for token, index in vocab.items():
                if token.startswith("▁") and token.split("▁")[-1].isalpha():  # TODO don't know what's this underline
                    token_list.append(token)
                    token_index_list.append(index)
                    if len(token_list) == src_soft_num + tgt_soft_num:
                        break
            # print(f"Shared tokens:\n{token_list[0: args.sha_num]}\n Unshared tokens:\n{token_list[args.sha_num:]}\n")
            src_embeds = raw_embedding.weight[torch.tensor(token_index_list[0: src_soft_num], dtype=torch.long)].clone()
            tgt_embeds = raw_embedding.weight[torch.tensor(token_index_list[0: tgt_soft_num], dtype=torch.long)].clone()
            assert len(src_embeds) == src_soft_num
            assert len(tgt_embeds) == tgt_soft_num
            assert len(src_embeds) + len(tgt_embeds) == len(token_index_list)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    src_embeds = torch.nn.Parameter(src_embeds, requires_grad=True)
    tgt_embeds = torch.nn.Parameter(tgt_embeds, requires_grad=True)
    return src_embeds, tgt_embeds


def get_data_handler(plm, WrapperClass, tokenizer, args, seed):
    raw_embedding = plm.get_input_embeddings()
    raw_embedding.requires_grad_(False)

    src_soft_embeds, tgt_soft_embeds = get_embeds(init_vocab=args.init_vocab, tokens_from=args.tokens_from,
                                                  src_soft_num=args.src_soft_num, tgt_soft_num=args.tgt_soft_num,
                                                  tokenizer=tokenizer, raw_embedding=raw_embedding)

    if args.ppt:
        ppt_path = f"{args.experiment_dir}/ppt-nsp/pretrain-nsp.pt"
        assert args.model_name == "t5-xxl-lm-adapt"
        ppt_embeds = torch.load(ppt_path, map_location=torch.device('cpu'))
        src_soft_embeds, tgt_soft_embeds = ppt_embeds.clone(), ppt_embeds.clone()


    # =========================== tgt data path ===========================
    if args.test and (args.test_data != "" and args.test_data != None):
        tgt_dataset_name = args.test_data
    else:
        tgt_dataset_name = args.tgt_data
    if args.tgt_full:
        data_path = os.path.join(args.data_dir, "original")
        tgt_dataset = get_data_setting(path=data_path, dataset_name=tgt_dataset_name)
    elif args.tgt_shot:
        data_path = os.path.join(args.data_dir, f"{args.shot_num}-shot")
        short_dir = f"{str(args.shot_num) + '-' + str(seed)}"
        tgt_dataset = get_few_shot_data_setting(path=data_path, dataset_name=tgt_dataset_name, short_dir=short_dir)
    else:
        raise NotImplementedError

    # =========================== tgttemplate ===========================
    # Note that soft template can be combined with hard template, by loading the hard template from file.
    # For example, the template in soft_template.txt is {} The choice_id 1 is the hard template
    """post_log_softmax=False to disable the process_logits function in manual_verbalizer.py which applied softmax over logits of the labels"""
    """NLLLoss should be used if it is set to be True"""

    # tgt verbalizer
    script_path = os.path.join(args.script_path, f"{tgt_dataset_name.split('_')[0]}")
    tgt_verbalizer = ManualVerbalizer(tokenizer, num_classes=args.num_classes, post_log_softmax=False).from_file(
        f"{script_path}/manual_verbalizer.txt")

    # tgt template
    tgt_template = MySoftTemplate(plm=plm, tokenizer=tokenizer, domain_num_tokens=args.tgt_soft_num,
                                  domain_soft_embeds=tgt_soft_embeds,
                                  initialize_from_vocab=args.init_vocab).from_file(f"{script_path}/soft_template.txt", choice=args.temp_id)

    # TODO be sure to use teacher_forcing and predict_eos_token=True  (Generation). predict_eos_token=False generates loss_ids only on label words

    # =========================== tgt tensor dataset ===========================

    train_dataset = MyPromptDataset(dataset=tgt_dataset["train"], template=tgt_template, verbalizer=tgt_verbalizer, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                    decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")
    val_dataset = MyPromptDataset(dataset=tgt_dataset["validation"], template=tgt_template, verbalizer=tgt_verbalizer, tokenizer=tokenizer,
                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                  decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")

    if args.tgt_shot:
        processed_path = os.path.join(args.experiment_dir, "processed_data", f"{args.shot_num}-shot", f"{str(args.shot_num) + '-' + str(seed)}")
    elif args.tgt_full:
        processed_path = os.path.join(args.experiment_dir, "processed_data", "full")
    else:
        raise NotImplementedError
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    train_file = os.path.join(processed_path, f"{tgt_dataset_name}.train")
    validation_file = os.path.join(processed_path, f"{tgt_dataset_name}.validation")
    test_file = os.path.join(processed_path, f"{tgt_dataset_name}.test")

    if not os.path.isfile(train_file):
        tgt_train_dataset = train_dataset.process()  # add manual template
        torch.save(tgt_train_dataset, train_file)
    else:
        tgt_train_dataset = torch.load(train_file)

    if not os.path.isfile(validation_file):
        tgt_validation_dataset = val_dataset.process()
        torch.save(tgt_validation_dataset, validation_file)
    else:
        tgt_validation_dataset = torch.load(validation_file)

    # tgt test
    if len(tgt_dataset["test"]) == 0:
        tgt_test_dataset = None
    else:
        dataset = MyPromptDataset(dataset=tgt_dataset["test"], template=tgt_template, verbalizer=tgt_verbalizer, tokenizer=tokenizer,
                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_len,
                                  decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")
        if not os.path.isfile(test_file):
            tgt_test_dataset = dataset.process()
            torch.save(tgt_test_dataset, test_file)
        else:
            tgt_test_dataset = torch.load(test_file)

    print(f"tgt data: {tgt_dataset_name}, seq_len:{args.max_seq_len} ( train:{len(tgt_train_dataset[0]['input_ids'])}, val:{len(tgt_validation_dataset[0]['input_ids'])})")


    ## ================================== src data path =================================

    if args.src_full:
        data_path = os.path.join(args.data_dir, "original")
        src_dataset = get_data_setting(path=data_path, dataset_name=args.src_data)
    else:
        data_path = os.path.join(args.data_dir, f"{args.shot_num}-shot")
        short_dir = f"{str(args.shot_num) + '-' + str(seed)}"
        src_dataset = get_few_shot_data_setting(path=data_path, dataset_name=args.src_data, short_dir=short_dir)

    # =========================== src template ===========================
    if args.src:
        script_path = os.path.join(args.script_path, f"{args.src_data.split('_')[0]}")
        if args.src_data in ["snli", "mnli_matched", "mnli_mismatched"]:
            src_num_cls = 3
            src_temp_id = 3
        else:
            src_num_cls = args.num_classes
            src_temp_id = args.temp_id
        src_verbalizer = ManualVerbalizer(tokenizer, num_classes=src_num_cls, post_log_softmax=False).from_file(
            f"{script_path}/manual_verbalizer.txt")

        src_template = MySoftTemplate(plm=plm, tokenizer=tokenizer, domain_num_tokens=args.src_soft_num,
                                      domain_soft_embeds=src_soft_embeds,
                                      initialize_from_vocab=args.init_vocab).from_file(f"{script_path}/soft_template.txt", choice=src_temp_id)

        if args.src_max_seq_len is not None:
            src_max_seq_len = args.src_max_seq_len
        else:
            src_max_seq_len = args.max_seq_len

        # =========================== src tensor dataset ===========================
        train_dataset = MyPromptDataset(dataset=src_dataset["train"], template=src_template, verbalizer=src_verbalizer, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=src_max_seq_len,
                                        decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")

        val_dataset = MyPromptDataset(dataset=src_dataset["validation"], template=src_template, verbalizer=src_verbalizer, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=src_max_seq_len,
                                      decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")

        train_file = os.path.join(processed_path, f"{args.src_data}.train")
        validation_file = os.path.join(processed_path, f"{args.src_data}.validation")
        test_file = os.path.join(processed_path, f"{args.src_data}.test")

        if not os.path.isfile(train_file):
            src_train_dataset = train_dataset.process()
            torch.save(src_train_dataset, train_file)
        else:
            src_train_dataset = torch.load(train_file)

        if not os.path.isfile(validation_file):
            src_validation_dataset = val_dataset.process()
            torch.save(src_validation_dataset, validation_file)
        else:
            src_validation_dataset = torch.load(validation_file)

        # src test
        if len(src_dataset["test"]) == 0:
            src_test_dataset = None
        else:
            dataset = MyPromptDataset(dataset=src_dataset["test"], template=src_template, verbalizer=src_verbalizer, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=src_max_seq_len,
                                      decoder_max_length=args.decoder_max_len, teacher_forcing=False, predict_eos_token=False, truncate_method="tail")
            if not os.path.isfile(test_file):
                src_test_dataset = dataset.process()
                torch.save(src_test_dataset, test_file)
            else:
                src_test_dataset = torch.load(test_file)

        print(f"src data: {args.src_data}, seq_len:{src_max_seq_len} ( train:{len(src_train_dataset[0]['input_ids'])}, val:{len(src_validation_dataset[0]['input_ids'])})")
    else:
        # useless, just for iterating on some data to satisfy the predefined training loop
        src_train_dataset, src_validation_dataset, src_test_dataset, src_template, src_verbalizer = tgt_train_dataset, tgt_validation_dataset, tgt_test_dataset, tgt_template, tgt_verbalizer
    return tgt_train_dataset, tgt_validation_dataset, tgt_test_dataset, tgt_template, tgt_verbalizer, \
           src_train_dataset, src_validation_dataset, src_test_dataset, src_template, src_verbalizer

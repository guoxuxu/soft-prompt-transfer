import copy
import os, shutil, time
from mymodules.myprompt_model import MyPromptModelForClassification
from mymodules.utils import format_time
from data_utils import get_data_handler
from mymodules.myprompt_loader import MyPromptDataLoader
from train_utils import get_domain_discriminator
from train_utils import get_optimizers
from evaluation import evaluate, predict

from mymodules.free import freeLB
from mymodules.optima import OPTIMA
from mymodules.vat import VAT

from openprompt.plms import load_plm
from openprompt.utils.reproduciblity import set_seed
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser("")
# Model
parser.add_argument("--model", type=str, default='t5-lm')
parser.add_argument("--model_name", default='t5-large-lm-adapt')
parser.add_argument("--decoder_max_len", type=int, default=3)
parser.add_argument("--src_max_seq_len", type=int)
parser.add_argument("--max_seq_len", type=int, default=100)
parser.add_argument("--eval", action="store_false", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune", action="store_true")
model_names = ["model_name", "decoder_max_len", "max_seq_len", "eval", "tune"]

# Prompt
parser.add_argument("--load_from_local", action="store_true")
parser.add_argument("--verb_type", type=str, default="manual", help="manual, generation")
parser.add_argument("--temp_id", type=int, default=0)
parser.add_argument("--src_soft_num", type=int, default=100)
parser.add_argument("--tgt_soft_num", type=int, default=100)
parser.add_argument("--init_vocab", action="store_false")
parser.add_argument("--tokens_from", type=str, default="firstWords", help="first, firstWords")

prompt_names = ["verb_type", "temp_id", "src_soft_num", "tgt_soft_num", "init_vocab", "tokens_from"]

# Data
parser.add_argument("--src_data", type=str, help="mr, snli, mrpc")
parser.add_argument("--src_full", action="store_true")
parser.add_argument("--src_shot", action="store_true")
parser.add_argument("--tgt_data", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--tgt_full", action="store_true")
parser.add_argument("--tgt_shot", action="store_true")
parser.add_argument("--shot_num", type=int, default=8, help="16, 64, 128")
parser.add_argument("--shot_seed", type=int, help="")
parser.add_argument("--num_classes", type=int, help="2, 3")
data_names = ["src_data", "src_full", "src_shot", "tgt_data", "tgt_full", "tgt_shot", "shot_num", "shot_seed", "num_classes"]

# Method
parser.add_argument("--src", action="store_true", help="src prompt is going to be trained")
parser.add_argument("--tgt", action="store_true", help="tgt prompt is going to be trained")
parser.add_argument("--ad", action="store_true", help="src input and tgt input is restricted to be invariant")
parser.add_argument("--ad_ramp", action="store_true")
parser.add_argument("--adlr", type=float)
parser.add_argument("--reload", action="store_true", help="reload the src prompt as the tgt prompt")
parser.add_argument("--load_seed", type=int)
#
parser.add_argument("--adv", action="store_true", help="apply adv perturbation, if false, turn back to normal training with ce loss")
parser.add_argument("--adv_method", type=str, help="the method to compute adv")
parser.add_argument("--adv_lr", type=float, default=0.1, help="")
parser.add_argument("--adv_steps", type=int, default=1, help="how many steps to adv")
parser.add_argument("--init_mag", type=int, default=1, help="init the sigma, 0->zero, >0 ->")
parser.add_argument("--norm_type", type=str, default="l2", help="l2, linf")
parser.add_argument("--adv_max_norm", type=float, default=0, help="set to 0 to be unlimited")
#
method_names = ["src", "tgt", "ad", "adlr", "ad_ramp", "reload"]
adv_method_names = ["adv", "adv_lr", "adv_method", "adv_steps", "init_mag", "norm_type", "adv_max_norm"]

# Training
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--plr", type=float, default=0.3)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--scheduler", type=str, default="cosine", help="linear, cosine")
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--max_steps", default=30000, type=int, help="20000")
parser.add_argument("--eval_every", type=int, default=1000)
training_names = ["optimizer", "plr", "scheduler", "wd", "batchsize", "max_steps", "eval_every"]
#
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--config", type=str, default="")
parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--tgt_unlabeled", action="store_false")
parser.add_argument("--seed", type=int)
parser.add_argument("--test", action="store_true")
parser.add_argument("--print_val", action="store_true")
parser.add_argument("--print_test", action="store_true")
parser.add_argument("--summarize_seeds", action="store_true")
parser.add_argument("--parallelize", action="store_true", help="if multiple gpus are available, one can use parallelize")
parser.add_argument("--seed_list", type=int, nargs="+")

parser.add_argument("--ppt", action="store_true")
parser.add_argument("--ppt_dir", type=str, default="../ppt-nsp/pretrain-nsp.pt")
parser.add_argument("--results_dir", type=str, default="./")
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--plm_path", type=str, default="../pretrained_models/")
parser.add_argument("--script_path", type=str, default="./scripts/")
args = parser.parse_args()
other_names = ["cuda", "cfg", "ckpt", "tgt_unlabeled", "seed", "test", "print_val", "print_test",
               "summarize_seeds", "script_path", "parallelize", "seed_list",
               "ppt", "ppt_dir", "results_dir", "data_dir", "plm_path", "script_path"]

args_copy = copy.deepcopy(args)

from utils import load_args, external_args

args = load_args(old_args=args, config_name=args.config)

## =============================== Check args ===============================

if args.eval and args.tune:
    raise NotImplementedError
if (args.src_full and args.src_shot) or (args.tgt_full and args.tgt_shot):
    raise NotImplementedError
if not (args.src_full or args.src_shot) and not (args.tgt_full or args.tgt_shot):
    raise NotImplementedError

seed_list = args.seed_list

import torch
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:" + str(args.cuda))

# ================================= Create File paths =================================

if (args.ckpt != "" or args.ppt) and args.tgt:
    args, ext_args = external_args(args, other_names, setting_name_list=["adv_lr", "adlr", "adxl", "max_steps", "adv_max_norm", "adv_steps", "init_mag", "batchsize"], skip_list=["tune", "eval"])
    if args.ppt:
        config_path = f"{args.config}/PPT"
    else:
        assert args.ckpt != ""
        src_config, src_data = args.ckpt.split("/")
        config_path = f"{args.config}/{src_config}_{'_'.join(ext_args)}_seed{args.load_seed}/{src_data}"
else:
    if args.tune:
        args = external_args(args, other_names, skip_list=["tune", "eval"])
    else:
        args = external_args(args, other_names)
    config_path = args.config

from utils import create_file_path

args_names_lists = [model_names, prompt_names, data_names, method_names, adv_method_names, training_names]
experiment_path = create_file_path(root_path=args.results_dir, config_path=config_path, all_vars=vars(args), names=args_names_lists)


## =============================== Summarize Performance across seeds ===============================
if args.summarize_seeds:
    from utils import summarize_seeds
    summarize_seeds(args, seed_list, experiment_path)


print(experiment_path)
## =============================== Seed for reproduce ===============================
all_tgt_val_acc, all_tgt_val_f1 = [], []

for seed in seed_list:
    set_seed(seed)
    content_write = "=" * 10

    # =============================== Create File paths ===============================
    results_save_path = os.path.join(experiment_path, f"seed{seed}")
    args.results_save_path = results_save_path
    ckpt_save_path = os.path.join(results_save_path, "ckpt")
    logs_save_path = os.path.join(results_save_path, "logs")
    this_run_result_file = os.path.join(results_save_path, f"summary.txt")

    if not args.test:
        writer = SummaryWriter(logs_save_path)  # writer creates path
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
    else:
        writer = None

    ## =============================== Load PLM ===============================
    if args.load_from_local:
        model_path = os.path.join(args.plm_path, args.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError
    else:
        # use lm-adapted version or t5-v1.1 checkpoint. Note that the originial t5 checkpoint has been pretrained on part of GLUE dataset, thus should not be used.
        model_path = args.model_name
        if "lm-adapt" in model_path:
            model_path = "google/" + model_path

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, model_path)  # TODO plm.encoder.embed_tokens.weight.requires_grad=False upon loaded

    ## =============================== Fetch the data and prepare dataloaders ===============================

    tgt_train_dataset, tgt_validation_dataset, tgt_test_dataset, tgt_template, tgt_verbalizer, \
    src_train_dataset, src_validation_dataset, src_test_dataset, src_template, src_verbalizer = get_data_handler(plm, WrapperClass, tokenizer, args, seed)

    if args.tune:
        plm.get_input_embeddings().weight.requires_grad_(True)

    assert len(tgt_train_dataset[0]["input_ids"]) == args.max_seq_len

    tgt_validation_dataloader = MyPromptDataLoader(dataset=tgt_validation_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=False)
    if tgt_test_dataset is None or len(tgt_test_dataset) == 0:
        tgt_test_dataloader = None
    else:
        tgt_test_dataloader = MyPromptDataLoader(dataset=tgt_test_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=False)

    src_validation_dataloader = MyPromptDataLoader(dataset=src_validation_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=False)
    if src_test_dataset is None:
        src_test_dataloader = None
    else:
        src_test_dataloader = MyPromptDataLoader(dataset=src_test_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=False)

    # =============================== Create Model and Domain Discriminator  ===============================
    if args.ad_ramp:
        domain_discriminator = get_domain_discriminator(plm=plm)
    else:
        domain_discriminator = None

    if domain_discriminator is not None:
        domain_discriminator = domain_discriminator.to(device)


    tgt_prompt_model = MyPromptModelForClassification(domain_disc=domain_discriminator, verbalizer=tgt_verbalizer, plm=plm,
                                                      template=tgt_template, freeze_plm=(not args.tune), plm_eval_mode=args.eval)

    src_prompt_model = MyPromptModelForClassification(domain_disc=domain_discriminator, verbalizer=src_verbalizer, plm=plm,
                                                      template=src_template, freeze_plm=(not args.tune), plm_eval_mode=args.eval)

    if args.parallelize:
        src_prompt_model.parallelize()
        tgt_prompt_model.parallelize()
    else:
        src_prompt_model = src_prompt_model.to(device)
        tgt_prompt_model = tgt_prompt_model.to(device)

    # =============================== Reload soft prompts from src ckpt to init tgt model ============================
    if args.reload:
        assert args.tgt
        if args.ppt:
            ppt_path = f"{args.ppt_dir}"
            assert args.model_name == "t5-xxl-lm-adapt"
            saved_state = torch.load(ppt_path, map_location=torch.device('cpu'))
            if seed == seed_list[0]:
                print(f"Reload from {ppt_path}")
            content_write += f"\n{ppt_path}\n"
            if args.ppt:
                tgt_prompt_model.template.unshared_soft_embeds.data = saved_state.to(device)
            else:
                filtered = {k: saved_state.get(k) for k in src_prompt_model.state_dict()}
                src_prompt_model.load_state_dict(filtered)
                filtered = {k: saved_state.get(k) for k in tgt_prompt_model.state_dict()}
                tgt_prompt_model.load_state_dict(filtered)

        if args.ckpt != "":
            src_args = load_args(old_args=args_copy, config_name=args.ckpt)
            src_args = external_args(src_args, other_names, skip_list=["reload"])
            src_experiment_path = create_file_path(root_path=args.results_dir, config_path=args.ckpt, all_vars=vars(src_args), names=args_names_lists)
            src_ckpt_file = os.path.join(src_experiment_path, f"seed{args.load_seed}", "ckpt", "bestStep.pt")
            if not os.path.isfile(src_ckpt_file):
                src_ckpt_file = os.path.join(src_experiment_path, f"seed{args.load_seed}", "ckpt", "init.pt")
                if not os.path.isfile(src_ckpt_file):
                    print(src_ckpt_file)
                    raise FileNotFoundError
            saved_state = torch.load(src_ckpt_file, map_location=torch.device('cpu'))
            if seed == seed_list[0]:
                print(f"Reload from {src_ckpt_file}")
            content_write += f"\n{src_ckpt_file}\n"

            filtered = {k: saved_state.get(k) for k in tgt_prompt_model.state_dict()}
            tgt_prompt_model.load_state_dict(filtered)

    # =============================== Test Performance ===============================
    if args.test:
        ckpt_file = os.path.join(ckpt_save_path, "bestStep.pt")
        if not os.path.isfile(ckpt_file):
            ckpt_file = os.path.join(ckpt_save_path, "init.pt")
            if not os.path.isfile(ckpt_file):
                seed_list.remove(seed)
                continue
        print(f"test model: {ckpt_file}")
        saved_state = torch.load(ckpt_file, map_location=torch.device('cpu'))

        if args.tgt:
            filtered = {k: saved_state.get(k) for k in tgt_prompt_model.state_dict()}
            tgt_prompt_model.load_state_dict(filtered)
        else:
            filtered = {k: saved_state.get(k) for k in src_prompt_model.state_dict()}
            src_prompt_model.load_state_dict(filtered)

        if args.tgt:
            eval_model = tgt_prompt_model
        else:
            src_prompt_model.verbalizer = tgt_verbalizer
            eval_model = src_prompt_model

        if args.tgt_full:
            # test tgt val
            tgt_val_acc, tgt_val_f1, tgt_val_loss, _ = evaluate(eval_model, writer, device, args, tgt_validation_dataloader, data_set="validation", domain="tgt")
        elif args.tgt_shot:
            # test tgt
            tgt_val_acc, tgt_val_f1, tgt_val_loss, _ = evaluate(eval_model, writer, device, args, tgt_test_dataloader, data_set="test", domain="tgt")
        else:
            raise NotImplementedError

        print(f"seed:{seed}, tgt_test_acc:{tgt_val_acc}, tgt_test_f1:{tgt_val_f1}")
        all_tgt_val_acc.append(tgt_val_acc)
        all_tgt_val_f1.append(tgt_val_f1)

        tmp = f"\nTgtTestACC:{tgt_val_acc}\t TgtTestF1:{tgt_val_f1}\n{format_time()}\n"

        with open(f"{this_run_result_file}", "a") as fout:
            fout.write(tmp)

        if args.parallelize:
            src_prompt_model.deparallelize()
            tgt_prompt_model.deparallelize()

        continue

    # =============================== Training ===============================

    tgt_plm_optimizer, tgt_plm_scheduler, tgt_prompt_optimizer, tgt_prompt_scheduler, \
    src_plm_optimizer, src_plm_scheduler, src_prompt_optimizer, src_prompt_scheduler, \
    domain_optimizer, domain_scheduler = get_optimizers(args, src_prompt_model, tgt_prompt_model)

    best_glb_step = 0
    best_val_acc = 0  #
    val_acc_traces = []

    # compute time
    avg_train_time = 0
    glb_step = 0
    leave_training = False

    # =============================== iterate once on the larger dataset ===============================
    pbar = tqdm(total=args.max_steps, desc=f"seed{seed}")
    for epoch in range(1000000):  # 1000000

        # shuffle dataloader every time
        tgt_train_dataloader = MyPromptDataLoader(dataset=tgt_train_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=True, drop_last=True)
        src_train_dataloader = MyPromptDataLoader(dataset=src_train_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=True, drop_last=True)

        src_train_iter = iter(src_train_dataloader)
        tgt_train_iter = iter(tgt_train_dataloader)
        max_data_loader_len = max(len(src_train_dataloader), len(tgt_train_dataloader))

        for step in range(0, max_data_loader_len):
            """if ad_ramp is True: apply ad_weight- exponentially increasing ad power (Ramp-up), ramp_pert: percent of the training for rampup"""

            if args.ad_ramp:
                ad_weight = 2. / (1. + np.exp(-10 * float(glb_step / args.max_steps))) - 1
            else:
                ad_weight = 1.

            # =============================== Evaluation every args.eval_every steps ===============================
            ckpt_name = "init.pt" if glb_step == 0 else "bestStep.pt"
            if glb_step == 0:
                content_write += f"\nRandom Init Performance: glb_step/max_steps:{glb_step}/{args.max_steps}\n"

            if glb_step % args.eval_every == 0:
                if args.src:
                    src_val_acc, src_val_f1, src_val_loss, src_ad_val_loss = evaluate(src_prompt_model, writer, device, args, src_validation_dataloader, data_set="validation", domain="src")
                    val_acc_traces.append(src_val_acc)
                    tmp = f"ValACC:{src_val_acc}, Val F1: {src_val_f1}\n"
                    writer.add_scalar(tag="src/val_acc", scalar_value=src_val_acc, global_step=glb_step)
                    writer.add_scalar(tag="src/val_f1", scalar_value=src_val_f1, global_step=glb_step)
                    writer.add_scalar(tag="src/val_loss", scalar_value=src_val_loss, global_step=glb_step)
                    if args.ad_ramp:
                        writer.add_scalar(tag="ad/src_val_loss", scalar_value=src_ad_val_loss, global_step=glb_step)

                    if src_val_acc > best_val_acc:
                        best_glb_step = glb_step
                        best_val_acc = src_val_acc
                        torch.save(src_prompt_model.state_dict(), os.path.join(ckpt_save_path, ckpt_name))
                        training_states = {
                            'glb_step': glb_step,
                            'best_glb_step': best_glb_step,
                            'best_val_acc': best_val_acc,
                            'val_acc_traces': val_acc_traces,
                            'src_prompt_optim': src_prompt_optimizer.state_dict(),
                            'src_prompt_sched': src_prompt_scheduler.state_dict()
                        }
                        torch.save(training_states, os.path.join(ckpt_save_path, "src_training_states.pt"))
                    pbar.set_postfix({'best_step': best_glb_step, 'best_acc': best_val_acc, "curr_acc": src_val_acc, 'time': format_time()})
                    if glb_step != 0:
                        pbar.update(args.eval_every)

                if args.tgt:
                    tgt_val_acc, tgt_val_f1, tgt_val_loss, tgt_ad_val_loss = evaluate(tgt_prompt_model, writer, device, args, tgt_validation_dataloader, data_set="validation", domain="tgt")
                    val_acc_traces.append(tgt_val_acc)
                    tmp = f"ValACC:{tgt_val_acc}, Val F1: {tgt_val_f1}\n"
                    writer.add_scalar(tag="tgt/val_acc", scalar_value=tgt_val_acc, global_step=glb_step)
                    writer.add_scalar(tag="tgt/val_f1", scalar_value=tgt_val_f1, global_step=glb_step)
                    writer.add_scalar(tag="tgt/val_loss", scalar_value=tgt_val_loss, global_step=glb_step)
                    if args.ad_ramp:
                        writer.add_scalar(tag="ad/tgt_val_loss", scalar_value=tgt_ad_val_loss, global_step=glb_step)

                    if glb_step % (args.max_steps / 10) == 0 and seed in seed_list[0:2]:
                        torch.save(tgt_prompt_model.state_dict(), os.path.join(ckpt_save_path, f"{glb_step}.pt"))
                    if tgt_val_acc > best_val_acc:
                        best_glb_step = glb_step
                        best_val_acc = tgt_val_acc
                        torch.save(tgt_prompt_model.state_dict(), os.path.join(ckpt_save_path, ckpt_name))

                    pbar.set_postfix({'best_step': best_glb_step, 'best_acc': best_val_acc, "curr_acc": tgt_val_acc, 'time': format_time()})
                    if glb_step != 0:
                        pbar.update(args.eval_every)

                if glb_step == 0:
                    content_write += tmp + "=" * 10   # init performance

                with open(f"{this_run_result_file}", "w") as fout:
                    fout.write(content_write)
                    fout.write(f"\nbest_glb_step/glb_steps:{best_glb_step}/{args.max_steps}\n")
                    fout.write(f"\nBestValACC:{best_val_acc}\n")

                if args.ad_ramp:
                    writer.add_scalar(tag="ad_weight", scalar_value=ad_weight, global_step=glb_step)

                src_prompt_model.train()
                tgt_prompt_model.train()

            # ===================== Early Stopping =====================
            if not (args.tgt_shot or args.src_shot):
                if (glb_step > args.max_steps * 0.3 and glb_step - best_glb_step >= args.max_steps * 0.3):  # early stop if no more better after * of total eval steps
                    leave_training = True
                    break
            if glb_step >= args.max_steps:
                leave_training = True
                break

            # =============================== cycle on the smaller dataset ===============================
            if glb_step> 0 and glb_step % len(src_train_dataloader) == 0 and len(src_train_dataloader) < len(tgt_train_dataloader):
                src_train_dataloader = MyPromptDataLoader(dataset=src_train_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=True, drop_last=True)
                src_train_iter = iter(src_train_dataloader)

            if glb_step> 0 and glb_step % len(tgt_train_dataloader) == 0 and len(tgt_train_dataloader) < len(src_train_dataloader):
                tgt_train_dataloader = MyPromptDataLoader(dataset=tgt_train_dataset, tokenizer=tokenizer, batch_size=args.batchsize, shuffle=True, drop_last=True)
                tgt_train_iter = iter(tgt_train_dataloader)

            src_inputs = next(src_train_iter)
            tgt_inputs = next(tgt_train_iter)

            for k, v in src_inputs.items():
                if type(v) is torch.Tensor:
                    src_inputs[k] = v.to(device)
            for k, v in tgt_inputs.items():
                if type(v) is torch.Tensor:
                    tgt_inputs[k] = v.to(device)

            time_start = time.perf_counter()
            # =============================== usual forward ===============================
            if not args.adv:  # this is for config of src_full

                if args.src or args.ad:
                    sum_loss = 0.

                    src_loss, src_logits, src_outputs = src_prompt_model(src_inputs)
                    writer.add_scalar(tag="src/train_loss", scalar_value=src_loss, global_step=glb_step)

                    if args.src:
                        sum_loss += src_loss  # src supervised

                    if args.ad:
                        domain_disc_src_loss, domain_disc_src_acc, disc_src_logits = src_prompt_model.domain_discrimination(outputs=src_outputs, ad_position=args.ad_pos, hidden_aggre=args.h_aggre, loss_ids=src_inputs["loss_ids"],
                                                                                device=src_loss.device, domain="src", ad_weight=ad_weight)
                        writer.add_scalar(tag="ad/src_train_loss", scalar_value=domain_disc_src_loss, global_step=glb_step)
                        sum_loss += domain_disc_src_loss * 0.5  # DANN

                    sum_loss.backward()

                if args.tgt or args.ad:  # this is for config of DANN_FULL
                    sum_loss = 0.

                    if args.src and not args.tgt:
                        this_model = src_prompt_model
                    else:
                        this_model = tgt_prompt_model

                    tgt_loss, tgt_logits, tgt_outputs = this_model(tgt_inputs)

                    if args.tgt:
                        sum_loss += tgt_loss  # tgt supervised
                        writer.add_scalar(tag="tgt/train_loss", scalar_value=tgt_loss, global_step=glb_step)

                    if args.ad:
                        domain_disc_tgt_loss, domain_disc_tgt_acc, disc_tgt_logits = this_model.domain_discrimination(outputs=tgt_outputs, ad_position=args.ad_pos, hidden_aggre=args.h_aggre, loss_ids=tgt_inputs["loss_ids"],
                                                                                device=tgt_loss.device, domain="tgt", ad_weight=ad_weight)
                        writer.add_scalar(tag="ad/tgt_train_loss", scalar_value=domain_disc_tgt_loss, global_step=glb_step)
                        sum_loss += domain_disc_tgt_loss * 0.5  # DANN

                    sum_loss.backward()


            # =============================== forward with AT ===============================
            else:
                if args.adv_method == "freeAT":  # ICLR 2020 FreeLB
                    freeLB(src_prompt_model, src_inputs, args, writer, glb_step)
                elif args.adv_method == "VAT":
                    VAT(src_prompt_model, src_inputs, args, writer, glb_step)
                elif args.adv_method == "optima":
                    OPTIMA(src_prompt_model, src_inputs, tgt_inputs, args, writer, glb_step)
                else:
                    raise NotImplementedError

            # =============================== model update ===============================
            if args.tgt and tgt_plm_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(plm.parameters(), 1.0)
                tgt_plm_optimizer.step()
                tgt_plm_optimizer.zero_grad()
                tgt_plm_scheduler.step()

            if args.src and src_plm_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(plm.parameters(), 1.0)
                src_plm_optimizer.step()
                src_plm_optimizer.zero_grad()
                src_plm_scheduler.step()

            if args.tgt and tgt_prompt_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(tgt_prompt_optimizer.param_groups[0]["params"], 1.0)
                tgt_prompt_optimizer.step()
                tgt_prompt_optimizer.zero_grad()
                tgt_prompt_scheduler.step()

            if args.src and src_prompt_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(src_prompt_optimizer.param_groups[0]["params"], 1.0)
                src_prompt_optimizer.step()
                src_prompt_optimizer.zero_grad()
                src_prompt_scheduler.step()

            if args.ad_ramp and domain_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(domain_optimizer.param_groups[0]["params"], 1.0)
                domain_optimizer.step()
                domain_optimizer.zero_grad()
                domain_scheduler.step()

            time_end = time.perf_counter()
            avg_train_time += (time_end - time_start)
            glb_step += 1

        if leave_training:
            break

    writer.close()

    if args.parallelize:
        src_prompt_model.deparallelize()
        tgt_prompt_model.deparallelize()

    content_write += f"\nbest_glb_step/glb_steps:{best_glb_step}/{args.max_steps},\tStop at {glb_step}\n"
    content_write += f"\nEnd Time {format_time()}\n average train time per step {avg_train_time / glb_step}\n"
    with open(f"{this_run_result_file}", "w") as fout:
        fout.write(content_write)

if args.test:
    print("=" * 10)
    print(f"seed: {seed_list}")
    print("=" * 10)
    ave_val_acc = torch.tensor(all_tgt_val_acc).mean().item()
    std_val_acc = torch.tensor(all_tgt_val_acc).std().item()
    var_val_acc = torch.tensor(all_tgt_val_acc).var().item()
    print(f"Tgt Test ACC: {all_tgt_val_acc} \nMean:{ave_val_acc:.2f}, Std: {std_val_acc:.2f}, var: {var_val_acc:.2f}")
    ave_val_f1 = torch.tensor(all_tgt_val_f1).mean().item()
    std_val_f1 = torch.tensor(all_tgt_val_f1).std().item()
    var_val_f1 = torch.tensor(all_tgt_val_f1).var().item()
    print(f"Tgt Test F1: {all_tgt_val_f1} \nMean:{ave_val_f1:.2f}, Std: {std_val_f1:.2f}, var: {var_val_f1:.2f}")
    print("=" * 10)
    print(f"{format_time()}")

import yaml, os, torch, sys


# ================================ load args =================================
def load_args(old_args, config_name):
    if "tgt_sup" in config_name:  # sup: supervised
        config_dir = "tgt_labeled_configs"
    else:
        config_dir = "tgt_unlabeled_configs"

    with open(os.path.join(config_dir, config_name + ".yml"), 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            vars(old_args).update(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)
    return old_args


def external_args(old_args, other_names, setting_name_list=None, skip_list=None):
    # ================================ overwrite config args with external commands =================================
    add_args = sys.argv
    sel_args = []
    for i in range(0, len(add_args)):
        arg = add_args[i]
        if arg.startswith("--"):
            arg = arg.split("--")[-1]

            if arg not in other_names:

                if type(vars(old_args)[arg]) is float:
                    value = float(add_args[i + 1])
                elif type(vars(old_args)[arg]) is str:
                    value = add_args[i + 1]
                elif type(vars(old_args)[arg]) is int:
                    value = int(add_args[i + 1])
                elif type(vars(old_args)[arg]) is bool:
                    value = True
                else:
                    raise NotImplementedError

                if skip_list is not None and arg in skip_list:
                    continue

                if setting_name_list is not None and arg in setting_name_list:
                    sel_args.append(f"{arg}{value}")
                    continue

                # only change those not in the lists
                vars(old_args)[arg] = value

    if setting_name_list is not None:
        return old_args, sel_args
    else:
        return old_args



# ================================= Create File paths =================================

def create_file_path(root_path, config_path, all_vars, names):
    model_names, prompt_names, data_names, method_names, adv_method_names, training_names = names
    data_prefix, method_prefix, adv_prefix, model_prefix, prompt_prefix, train_prefix = [], [], [], [], [], []
    for k, v in all_vars.items():
        if v is not None:
            s = str(k) + str(v)
            if k in model_names:
                model_prefix.append(s)
            if k in prompt_names:
                prompt_prefix.append(s)
            if k in data_names:
                if k == "shot_seed" and not (all_vars["tgt_shot"] or all_vars["src_shot"]):
                    continue
                if k == "shot_num" and not (all_vars["tgt_shot"] or all_vars["src_shot"]):
                    continue
                data_prefix.append(s)
            if k in method_names:
                method_prefix.append(s)
            if k in adv_method_names:
                if k == "ad_step" and v == 1:
                    continue
                adv_prefix.append(s)
            if k in training_names:
                train_prefix.append(s)

    if all_vars["adv"]:
        return os.path.join(root_path, config_path, "-".join(data_prefix), "-".join(method_prefix), "-".join(adv_prefix), "-".join(model_prefix),
                            "-".join(prompt_prefix), "-".join(train_prefix))
    else:
        return os.path.join(root_path, config_path, "-".join(data_prefix), "-".join(method_prefix), "-".join(model_prefix), "-".join(prompt_prefix),
                            "-".join(train_prefix))


## =============================== Summarize Performance across seeds ===============================
import ast
def summarize_seeds(args, seed_list, experiment_path):
    glb_steps, val_accs, init_val_accs, init_val_f1s, test_accs, test_f1s, test_train_accs, test_train_f1s, critical_steps = [], [], [], [], [], [], [], [], []
    for seed in seed_list:
        this_run_result_file = os.path.join(experiment_path, f"seed{seed}", f"summary.txt")
        if not os.path.isfile(this_run_result_file):
            seed_list.remove(seed)
            continue
        with open(this_run_result_file, encoding='utf-8') as f:
            lines = f.readlines()
            init_flag = 0
            for idx, line in enumerate(lines):
                if line.startswith("Random Init"):
                    init_flag = idx
                if idx == (init_flag + 1) and "ValACC" in line:
                    linelist = line.strip().split(',')
                    val = linelist[0].split(":")[-1]
                    init_val_accs.append(float(val))
                if idx == (init_flag + 1) and "Val F1" in line:
                    linelist = line.strip().split(',')
                    val = linelist[1].split(":")[-1]
                    init_val_f1s.append(float(val))
                if "best_glb_step/glb_steps" in line:
                    linelist = line.strip().split('\t')
                    glb_step = linelist[0].split(":")[-1].split("/")[0]
                    glb_steps.append(int(glb_step))
                if line.startswith("BestValACC"):
                    linelist = line.strip().split('\t')
                    val_acc = linelist[0].split(":")[-1]
                    val_accs.append(float(val_acc))

                if "TgtTestACC" in line:
                    linelist = line.strip().split('\t')
                    test_acc = linelist[0].split(":")[-1]
                    test_accs.append(float(test_acc))
                if "TgtTestF1" in line:
                    linelist = line.strip().split('\t')
                    test_f1 = linelist[1].split(":")[-1]
                    test_f1s.append(float(test_f1))


    if len(val_accs) == 0:
        print(experiment_path)
        print(os.listdir("/".join(experiment_path.split("/")[:-1])))
        exit()

    print(f" {'='*20}\nseed {seed_list}")

    if args.print_val:
        ave_val_acc = torch.tensor(val_accs).mean().item()
        std_val_acc = torch.tensor(val_accs).std().item()
        var_val_acc = torch.tensor(val_accs).var().item()
        print(f"steps {glb_steps}\nval_acc:{val_accs}\n Mean: {ave_val_acc:.2f} (std:{std_val_acc:.2f}, var:{var_val_acc:.2f})\n")

    if args.print_test:
        ave_test_acc = torch.tensor(test_accs).mean().item()
        std_test_acc = torch.tensor(test_accs).std().item()
        var_test_acc = torch.tensor(test_accs).var().item()
        print(f"test_accs:{test_accs}\n Mean: {ave_test_acc:.2f} (std:{std_test_acc:.2f}, var:{var_test_acc:.2f})\n")
        ave_test_f1 = torch.tensor(test_f1s).mean().item()
        std_test_f1 = torch.tensor(test_f1s).std().item()
        var_test_f1 = torch.tensor(test_f1s).var().item()
        print(f"test_f1s:{test_f1s}\n Mean: {ave_test_f1:.2f} (std:{std_test_f1:.2f}, var:{var_test_f1:.2f})\n")


    exit()



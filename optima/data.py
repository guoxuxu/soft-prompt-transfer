from datasets import concatenate_datasets, load_dataset
import numpy as np
import os, torch

# dataset = load_dataset("paws", "labeled_final")

# ===================== set

dataset_name = "cb"
short_num = 16
seed_list = [100, 13, 21, 42, 87, 36, 52, 68, 78, 93, 234, 63, 527, 628, 29, 15]


# ===================== begin
if dataset_name == "sick":
    dataset = load_dataset("sick")
elif dataset_name == "multirc":
    dataset = load_dataset("super_glue", "multirc")
    path = f"../../hdd/LM-BFF/data/{short_num}-shot/MultiRC"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(dataset["validation"], os.path.join(path, "test.pt"))
elif dataset_name == "cb":
    dataset = load_dataset("super_glue", "cb")
    path = f"../../hdd/LM-BFF/data/{short_num}-shot/CB"
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(dataset["validation"], os.path.join(path, "test.pt"))
else:
    raise NotImplemented


relatednesses = []
for seed in seed_list:
    if dataset_name == "cb":
        path = f"../../hdd/LM-BFF/data/{short_num}-shot/CB/{short_num}-{seed}"
        if not os.path.exists(path):
            os.makedirs(path)
        class_0 = dataset["train"].filter(lambda example: example["label"] == 0)
        class_1 = dataset["train"].filter(lambda example: example["label"] == 1)
        class_2 = dataset["train"].filter(lambda example: example["label"] == 2)
        class_0 = class_0.shuffle(seed=seed)
        class_1 = class_1.shuffle(seed=seed)
        class_2 = class_2.shuffle(seed=seed)

        if short_num >= 16:
            indices = np.random.choice(np.arange(0, len(class_2)/2), short_num)
            train_dataset = concatenate_datasets([class_0.select(np.arange(0, short_num)), class_1.select(np.arange(0, short_num)),
                                                  class_2.select(indices)])
            indices = np.random.choice(np.arange(len(class_2)/2, len(class_2)), short_num)
            val_dataset = concatenate_datasets([class_0.select(np.arange(short_num, short_num * 2)),
                                                class_1.select(np.arange(short_num, short_num * 2)),
                                                class_2.select(indices)])
        else:
            train_dataset = concatenate_datasets([class_0.select(np.arange(0, short_num)), class_1.select(np.arange(0, short_num)), class_2.select(np.arange(0, short_num))])
            val_dataset = concatenate_datasets([class_0.select(np.arange(short_num, short_num * 2)), class_1.select(np.arange(short_num, short_num * 2)), class_2.select(np.arange(short_num, short_num * 2))])
        torch.save(train_dataset, f"{path}/train.pt")
        torch.save(val_dataset, f"{path}/dev.pt")

    elif dataset_name == "sick":
        path = f"../../hdd/LM-BFF/data/{short_num}-shot/SICK/{short_num}-{seed}"
        if not os.path.exists(path):
            os.makedirs(path)
        class_0 = dataset["train"].filter(lambda example: example["label"]==0)
        class_1 = dataset["train"].filter(lambda example: example["label"]==1)
        class_2 = dataset["train"].filter(lambda example: example["label"]==2)
        class_0 = class_0.shuffle(seed=seed)
        class_1 = class_1.shuffle(seed=seed)
        class_2 = class_2.shuffle(seed=seed)
        train_dataset = concatenate_datasets([class_0.select(np.arange(0, short_num)), class_1.select(np.arange(0, short_num)), class_2.select(np.arange(0, short_num))])
        train_dataset.to_csv(f"{path}/train.tsv", sep="\t")
        val_dataset = concatenate_datasets([class_0.select(np.arange(short_num, short_num * 2)), class_1.select(np.arange(short_num, short_num * 2)), class_2.select(np.arange(short_num, short_num * 2))])
        val_dataset.to_csv(f"{path}/dev.tsv", sep="\t")
        dataset["test"].to_csv(f"{path}/test.tsv", sep="\t")
    else:
        raise NotImplemented


#
# print()
# print(f"min: {min(test['relatedness_score']):.2f}, max:{max(test['relatedness_score']):.2f}, "
#       f"mean:{np.array(test['relatedness_score']).mean():.2f}, std: {np.array(test['relatedness_score']).std():.2f}")


import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5


def get_module_list(num_layers, embedding_size):
    return torch.nn.ModuleList([torch.nn.Linear(embedding_size, 2, bias=True) for _ in range(0, num_layers)])

def get_domain_discriminator(plm):
    embedding_size = plm.encoder.embed_tokens.weight.size(1)  # T5
    domain_discriminator = get_module_list(num_layers=1, embedding_size=embedding_size)
    return domain_discriminator


# ============================ define lr scheduler ============================

def get_lr_scheduler(optimizer, args):
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.max_steps)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.max_steps)
    elif args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
    else:
        raise NotImplementedError
    return scheduler


# ============================ Warp optimizers ============================

def get_optimizers(args, src_prompt_model, tgt_prompt_model):

    # ============================ plm optimizer ============================
    no_decay = ['bias', 'LayerNorm.weight']  # it's always good practice to set no decay to biase and LayerNorm parameters
    if args.tgt and args.tune:  # normally we freeze the model when using soft_template. However, we keep the option to tune plm
        """tune use AdamW"""
        plm_parameters = [
            {'params': [p for n, p in tgt_prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in tgt_prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        tgt_plm_optimizer = AdamW(plm_parameters, lr=3e-5)
        tgt_plm_scheduler = get_lr_scheduler(tgt_plm_optimizer, args)
    else:
        tgt_plm_optimizer = None
        tgt_plm_scheduler = None

    if args.src and args.tune:  # normally we freeze the model when using soft_template. However, we keep the option to tune plm
        """tune use AdamW"""
        plm_parameters = [
            {'params': [p for n, p in src_prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
            {'params': [p for n, p in src_prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        src_plm_optimizer = AdamW(plm_parameters, lr=3e-5)
        src_plm_scheduler = get_lr_scheduler(src_plm_optimizer, args)
    else:
        src_plm_optimizer = None
        src_plm_scheduler = None
    # ============================ domain discriminator optimizer ============================
    if not args.adlr:
        adlr = args.plr
    else:
        adlr = args.adlr

    if args.ad_ramp:
        domain_parameters = [
            {'params': [p for n, p in src_prompt_model.domain_disc.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': args.wd, "lr": adlr},
            {'params': [p for n, p in src_prompt_model.domain_disc.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr":adlr},
        ]
        domain_optimizer = AdamW(domain_parameters, lr=args.plr)
        domain_scheduler = get_lr_scheduler(domain_optimizer, args)
    else:
        domain_optimizer = None
        domain_scheduler = None

    # ============================ prompt optimizer ============================

    tgt_prompt_parameters = [
        {'params': [p for n, p in tgt_prompt_model.template.named_parameters() if 'raw_embedding' not in n], 'weight_decay': args.wd}
    ]

    if args.optimizer.lower() == "adafactor":
        tgt_prompt_optimizer = Adafactor(tgt_prompt_parameters, lr=args.plr, relative_step=False, scale_parameter=False, warmup_init=False)
    elif args.optimizer.lower() == "adamw":
        tgt_prompt_optimizer = AdamW(tgt_prompt_parameters, lr=args.plr)
    else:
        raise NotImplementedError

    tgt_prompt_scheduler = get_lr_scheduler(tgt_prompt_optimizer, args)

    src_prompt_parameters = [
        {'params': [p for n, p in src_prompt_model.template.named_parameters() if 'raw_embedding' not in n], 'weight_decay': args.wd}
    ]

    if args.optimizer.lower() == "adafactor":
        # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        src_prompt_optimizer = Adafactor(src_prompt_parameters, lr=args.plr, relative_step=False, scale_parameter=False, warmup_init=False)
    elif args.optimizer.lower() == "adamw":
        src_prompt_optimizer = AdamW(src_prompt_parameters, lr=args.plr)  # usually lr = 0.5
    else:
        raise NotImplementedError

    src_prompt_scheduler = get_lr_scheduler(src_prompt_optimizer, args)

    return tgt_plm_optimizer, tgt_plm_scheduler, tgt_prompt_optimizer, tgt_prompt_scheduler,  \
           src_plm_optimizer, src_plm_scheduler, src_prompt_optimizer, src_prompt_scheduler, domain_optimizer, domain_scheduler



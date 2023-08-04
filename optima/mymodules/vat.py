import torch
import torch.nn.functional as F


"""
regarding the public VAT implementations in github:
This is implemented in TF by the VAT paper author: https://github.com/takerum/vat_tf/blob/master/vat.py
This one in PyTorch does not include self-entropy: https://github.com/lyakaap/VAT-pytorch/blob/master/train_CIFAR10.py
This one in PyTorch does include: https://github.com/9310gaurav/virtual-adversarial-training/blob/master/main.py
"""


def entropy_x(logit):
    p = F.softmax(logit, dim=1)
    return - (p * F.log_softmax(logit, dim=1)).sum(dim=1).mean()


def VAT(model, src_batch, args, writer, glb_step, self_entropy=False):

    # (+Regularization) VAT loss (0) clone the initial input_embeds
    raw_embedding = model.plm.get_input_embeddings()
    src_input_ids = src_batch['input_ids'].clone()
    src_mask = src_batch['attention_mask'].clone()
    src_embeds_init = raw_embedding(src_input_ids)
    src_input_batch = model.input_ids_to_embeds(src_batch)

    # ========================= sup loss =========================
    src_sup_loss, src_true_logits, outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
    src_sup_loss.backward()
    src_true_logits = src_true_logits.detach().clone()

    # print(f"{glb_step}: src cls loss:{src_sup_loss:.6f}, grad:{model.template.unshared_soft_embeds.grad.norm():.6f}")

    writer.add_scalar(tag="src/train_loss", scalar_value=src_sup_loss, global_step=glb_step)

    # ========================= init delta =========================
    if args.init_mag > 0:

        if args.norm_type == "l2":
            input_lengths = torch.sum(src_mask, 1)  #
            delta = torch.zeros_like(src_embeds_init).uniform_(-args.init_mag, args.init_mag) * src_mask.unsqueeze(2)
            dims = input_lengths * src_embeds_init.size(-1)
            mag = args.init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()

        elif args.norm_type == "linf":
            delta = torch.zeros_like(src_embeds_init).uniform_(-args.init_mag, args.init_mag) * src_mask.unsqueeze(2)
        else:
            raise NotImplementedError

    else:
        delta = torch.zeros_like(src_embeds_init)

    # Find delta_adv
    for astep in range(args.adv_steps):

        # ========================= add delta to src data =========================
        delta.requires_grad_()
        src_input_batch["inputs_embeds"] = delta + src_embeds_init
        src_input_batch["attention_mask"] = src_mask

        # ========================= src kl loss =========================
        loss, src_logits, outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
        src_kl_loss = F.kl_div(F.log_softmax(src_logits, 1), F.softmax(src_true_logits, 1), reduction="batchmean")

        # (4) get gradient on delta
        delta_grad = torch.autograd.grad(src_kl_loss, delta)[0].detach().clone()


        # ========================= normalize delta =========================
        if args.norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)   # Euclidian norm: \sqrt(sum(x_i^2))
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(src_embeds_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()

        elif args.norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            if args.adv_max_norm > 0:
                delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()

        src_embeds_init = raw_embedding(src_input_ids)


    # ========================= add found delta to src data =========================
    src_input_batch["inputs_embeds"] = delta + src_embeds_init
    src_input_batch["attention_mask"] = src_mask
    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).mean()
    writer.add_scalar(tag="adv/vat_src_delta", scalar_value=delta_norm, global_step=glb_step)


    # ========================= src kl loss =========================
    loss, src_logits, outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
    src_kl_loss = F.kl_div(F.log_softmax(src_logits, 1), F.softmax(src_true_logits, 1), reduction="batchmean")
    tot_loss = src_kl_loss
    writer.add_scalar(tag="src/kl_loss", scalar_value=src_kl_loss, global_step=glb_step)

    if self_entropy:
        src_ent_loss = entropy_x(src_logits)
        tot_loss += src_ent_loss
        writer.add_scalar(tag="src/ent_loss", scalar_value=src_ent_loss, global_step=glb_step)

    tot_loss.backward()





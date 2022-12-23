import torch, os
import torch.nn.functional as F


def entropy_x(logit):
    p = F.softmax(logit, dim=1)
    return - (p * F.log_softmax(logit, dim=1)).sum(dim=1).mean()


def OPTIMA(model, src_batch, tgt_batch, args, writer, glb_step):

    # (+Regularization) VAT loss (0) clone the initial input_embeds
    raw_embedding = model.plm.get_input_embeddings()
    src_input_ids = src_batch['input_ids'].clone()
    src_mask = src_batch['attention_mask'].clone()
    src_embeds_init = raw_embedding(src_input_ids)
    src_input_batch = model.input_ids_to_embeds(src_batch)

    # tgt batch
    tgt_input_ids = tgt_batch['input_ids'].clone()
    tgt_mask = tgt_batch['attention_mask'].clone()
    tgt_input_batch = model.input_ids_to_embeds(tgt_batch)

    # =============== train domain discriminator ===
    for i in range(0, 1):
        src_loss, src_logits, src_outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
        domain_disc_src_loss, domain_disc_src_acc = model.domain_discrimination(outputs=src_outputs, loss_ids=src_batch["loss_ids"],
                                                           device=src_loss.device, domain="src", ad_weight=0)  # ad_weight: disable gradient reversal
        domain_disc_src_loss.backward()
        tgt_loss, tgt_logits, tgt_outputs = model.forward_computation(tgt_input_batch, tgt_batch["loss_ids"], tgt_batch["label"])
        domain_disc_tgt_loss, domain_disc_tgt_acc = model.domain_discrimination(outputs=tgt_outputs, loss_ids=tgt_batch["loss_ids"],
                                                           device=tgt_loss.device, domain="tgt", ad_weight=0)

        domain_disc_tgt_loss.backward()
        src_embeds_init = raw_embedding(src_input_ids)
        tgt_embeds_init = raw_embedding(tgt_input_ids)
        src_input_batch["inputs_embeds"] = src_embeds_init
        src_input_batch["attention_mask"] = src_mask
        tgt_input_batch["inputs_embeds"] = tgt_embeds_init
        tgt_input_batch["attention_mask"] = tgt_mask


    # ========================= sup loss =========================
    src_sup_loss, src_true_logits, outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
    src_sup_loss.backward()
    src_true_logits = src_true_logits.detach().clone()
    writer.add_scalar(tag="src/train_loss", scalar_value=src_sup_loss, global_step=glb_step)


    # ========================= init delta =========================
    if args.init_mag > 0:

        if args.norm_type == "l2":
            input_lengths = torch.sum(src_mask, 1)
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


    for astep in range(args.adv_steps):

        # ========================= add delta on src data =========================
        delta.requires_grad_()
        src_input_batch["inputs_embeds"] = delta + src_embeds_init
        src_input_batch["attention_mask"] = src_mask

        # ========================= src kl loss + ad loss  =========================
        src_loss, src_logits, src_outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
        src_kl_loss = F.kl_div(F.log_softmax(src_logits, 1), F.softmax(src_true_logits, 1), reduction="batchmean")
        domain_disc_src_loss, domain_disc_src_acc = model.domain_discrimination(outputs=src_outputs, loss_ids=src_batch["loss_ids"],
                                                           device=src_loss.device, domain="src", ad_weight=1)

        sum_loss = src_kl_loss + domain_disc_src_loss * 0.5  # DANN

        # (4) get gradient on delta
        delta_grad = torch.autograd.grad(sum_loss, delta)[0].detach().clone()

        # print(f"delta:{delta.norm():.6f}, src_kl_loss:{src_kl_loss:.6f}, domain_disc_src_loss:{domain_disc_src_loss:.6f}")

        # ========================= normliaze delta =========================
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


    # ========================= add final delta on src =========================
    src_input_batch["inputs_embeds"] = delta + src_embeds_init
    src_input_batch["attention_mask"] = src_mask
    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).mean()
    writer.add_scalar(tag="adv/vat_src_delta", scalar_value=delta_norm, global_step=glb_step)

    # ========================= src kl loss =========================
    src_loss, src_logits, src_outputs = model.forward_computation(src_input_batch, src_batch["loss_ids"], src_batch["label"])
    src_kl_loss = F.kl_div(F.log_softmax(src_logits, 1), F.softmax(src_true_logits, 1), reduction="batchmean")
    domain_disc_src_loss, domain_disc_src_acc = model.domain_discrimination(outputs=src_outputs, loss_ids=src_batch["loss_ids"],
                                                                            device=src_loss.device, domain="src", ad_weight=0)
    tot_loss = src_kl_loss + domain_disc_src_loss * 0.5

    writer.add_scalar(tag="src/src_kl_loss", scalar_value=src_kl_loss, global_step=glb_step)

    # follow the practice in VAT author's repo
    src_ent_loss = entropy_x(src_logits)
    tot_loss += src_ent_loss

    tot_loss.backward()





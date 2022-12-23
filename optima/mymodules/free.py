import torch
"""
https://github.com/zhuchen03/FreeLB/blob/master/huggingface-transformers/examples/run_glue_freelb.py#L225
"""

def freeLB(model, batch, args, writer, glb_step):
    raw_embedding = model.plm.get_input_embeddings()
    batch_input_ids = batch['input_ids'].clone()
    batch_mask = batch['attention_mask'].clone()
    embeds_init = raw_embedding(batch_input_ids)

    input_batch = model.input_ids_to_embeds(batch)  # input_ids 8 * 100 -> input embeds 8 * 100 * 1024

    # ========================= init delta =========================

    if args.init_mag > 0:

        input_mask = batch['attention_mask'].to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        if args.norm_type == "l2":
            delta = torch.zeros_like(embeds_init).uniform_(-args.init_mag, args.init_mag) * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = args.init_mag / torch.sqrt(dims)  #
            delta = (delta * mag.view(-1, 1, 1)).detach()

        elif args.norm_type == "linf":
            delta = torch.zeros_like(embeds_init).uniform_(-args.init_mag, args.init_mag) * input_mask.unsqueeze(2)

        else:
            raise NotImplementedError

    else:
        delta = torch.zeros_like(embeds_init)

    assert args.adv_steps > 1

    for astep in range(args.adv_steps):
        # ========================= add delta =========================

        delta.requires_grad_()

        input_batch["inputs_embeds"] = delta + embeds_init  # 8 * 100 * 1024 -> 8 * 100 * 1024
        input_batch["attention_mask"] = batch_mask  # 8 * 100

        loss, label_words_logits, outputs = model.forward_computation(input_batch, batch["loss_ids"], batch["label"])  # concat prompt: 8 * 100 * 1024 -> 8 * 200 * 1024

        # (1) CE backward
        loss.backward()

        # (2) get gradient on delta
        delta_grad = delta.grad.clone().detach()

        # ========================= normalize delta =========================
        if args.norm_type == "l2":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)   # Euclidian norm: \sqrt(sum(x_i^2))
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            if args.adv_max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                delta = (delta * reweights).detach()

        elif args.norm_type == "linf":
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
            denorm = torch.clamp(denorm, min=1e-8)
            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

            if args.adv_max_norm > 0:
                delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()

        embeds_init = raw_embedding(batch_input_ids)

        # ========================= add log =========================
        if astep == 0:
            writer.add_scalar(tag="src/train_loss", scalar_value=loss, global_step=glb_step)
        elif astep == args.adv_steps - 1:
            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).mean()
            writer.add_scalar(tag="src/flb_src_delta", scalar_value=delta_norm, global_step=glb_step)
            writer.add_scalar(tag="src/flb_train_loss", scalar_value=loss, global_step=glb_step)


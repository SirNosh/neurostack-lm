from __future__ import annotations

import torch
import torch.nn.functional as F


def full_sequence_labels(
    input_ids: torch.Tensor, prompt_lengths: torch.Tensor
) -> torch.Tensor:
    labels = input_ids.clone()
    for row, length in enumerate(prompt_lengths.tolist()):
        labels[row, : int(length)] = -100
    return labels


def full_sequence_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, prompt_lengths: torch.Tensor
) -> torch.Tensor:
    labels = full_sequence_labels(input_ids, prompt_lengths)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def sequence_nll_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    *,
    prefix_length: int = 0,
) -> torch.Tensor:
    labels = torch.full(
        logits.shape[:2], -100, device=logits.device, dtype=torch.long
    )
    labels[:, prefix_length : prefix_length + input_ids.shape[1]] = input_ids
    for row, length in enumerate(prompt_lengths.tolist()):
        labels[row, : prefix_length + int(length)] = -100
    token_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(logits.shape[0], -1)
    valid = labels[:, 1:].ne(-100)
    return (token_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def fact_support_loss(
    fact_logits: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    margin: float = 0.2,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones_like(support_mask, dtype=torch.bool)
    fact_logits = fact_logits.masked_fill(~valid_mask, -1e4)
    target = support_mask / support_mask.sum(-1, keepdim=True).clamp_min(1)
    listwise = -(target * F.log_softmax(fact_logits, dim=-1)).sum(-1).mean()
    pair_losses = []
    for scores, mask, valid in zip(
        fact_logits, support_mask.bool(), valid_mask.bool()
    ):
        positive, negative = scores[mask & valid], scores[(~mask) & valid]
        if positive.numel() and negative.numel():
            pair_losses.append(
                F.softplus(margin - positive[:, None] + negative[None, :]).mean()
            )
    pairwise = torch.stack(pair_losses).mean() if pair_losses else listwise.new_zeros(())
    return listwise + 0.5 * pairwise


def distinct_support_slots(support_mask: torch.Tensor, slots: int = 8) -> torch.Tensor:
    targets = torch.full_like(support_mask, -100, dtype=torch.long)
    for row, mask in enumerate(support_mask.bool()):
        rank = 0
        for fact in mask.nonzero(as_tuple=False).flatten().tolist():
            targets[row, fact] = rank % slots
            rank += 1
    return targets

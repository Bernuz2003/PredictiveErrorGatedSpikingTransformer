from __future__ import annotations

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def timestep_accuracy(timestep_logits: torch.Tensor, target: torch.Tensor) -> list[float]:
    return [accuracy(timestep_logits[t], target, (1,))[0].item() for t in range(timestep_logits.shape[0])]


def confidence_entropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    conf = probs.max(dim=-1).values
    entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1)
    return conf, entropy


def confusion_matrix_rows(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[dict[str, int]]:
    mat = torch.zeros(num_classes, num_classes, dtype=torch.long, device=pred.device)
    for true, guess in zip(target.view(-1), pred.view(-1), strict=False):
        true_i = int(true.item())
        guess_i = int(guess.item())
        if 0 <= true_i < num_classes and 0 <= guess_i < num_classes:
            mat[true_i, guess_i] += 1
    rows = []
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(mat[true_label, pred_label].item()),
                }
            )
    return rows

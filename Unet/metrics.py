import torch

@torch.no_grad()
def seg_metrics(logits, target, threshold=0.5, eps=1e-7):
    """
    logits: (N,1,H,W) raw scores
    target: (N,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    pred  = (probs > threshold).float()

    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
    }

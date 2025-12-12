import os, torch

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_best(model, epoch, val_miou, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{model_name}_best_e{epoch:03d}_miou{val_miou:.4f}.pth"
    path = os.path.join(out_dir, fname)
    torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
    return path

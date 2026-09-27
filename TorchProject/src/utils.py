"""
Checkpoint utilities.

A checkpoint here means the FULL training state — not just the model's
weights — so training can be resumed exactly where it left off, or the
model can be loaded standalone for evaluation/inference. It also carries
the epoch-by-epoch training history, so plots after a resume still cover
the entire run, not just the epochs since the resume.
"""

import torch


def save_checkpoint(model, optimizer, scheduler, epoch: int,
                     best_metric: float, checkpoint_path: str,
                     history: dict = None) -> None:
    """
    Save everything needed to either:
    (a) resume training from this exact point, or
    (b) load just the model weights later for evaluation/inference.

    Args:
        model: the nn.Module being trained
        optimizer: the optimizer instance (its internal momentum/variance
                   buffers get saved too)
        scheduler: the LR scheduler instance, or None if not using one
        epoch: which epoch this checkpoint was taken at (so resume knows
               where to continue)
        best_metric: the best validation metric seen so far (so the
                     "is this better?" comparison stays correct after resume)
        checkpoint_path: file path to save to (e.g. "checkpoints/best.pth")
        history: dict of per-epoch lists (train_loss, val_loss, etc.),
                 saved so plots after a resume cover the full run
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model, optimizer=None,
                     scheduler=None, device: torch.device = None) -> dict:
    """
    Load a checkpoint back into model (and optionally optimizer/scheduler).

    Two usage modes:
      1. Resume training: pass optimizer AND scheduler so their internal
         state is restored too.
      2. Evaluation / inference only: pass optimizer=None, scheduler=None
         (default) — only the model weights get loaded.

    Returns the raw checkpoint dict, so the caller can read
    checkpoint["epoch"], checkpoint["best_metric"], and
    checkpoint["history"] if needed.

    `map_location=device` matters when the checkpoint was saved on a
    different device than the one you're loading on (e.g. saved on GPU,
    loading on this CPU-only laptop) — without it, torch.load raises an
    error trying to find a CUDA device that isn't there.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
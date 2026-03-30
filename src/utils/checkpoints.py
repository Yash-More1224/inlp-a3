import os
from pathlib import Path

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    
    # Handle backward compatibility: old checkpoints use "rnn.cell.*", new models use "rnn_layers.0.cell.*"
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("rnn.cell."):
            # Map old key to new key
            new_key = key.replace("rnn.cell.", "rnn_layers.0.cell.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {"epoch": checkpoint["epoch"], "loss": checkpoint["loss"]}

"""Shared configuration — device selection and loss function registry."""

import torch

# ─── Device ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Loss functions ─────────────────────────────────────────────────
criterion_dict = {
    'MSELoss': torch.nn.MSELoss(),
    'RMSELoss': torch.nn.MSELoss(),   # RMSE is applied via torch.sqrt() after
    'L1Loss': torch.nn.L1Loss(),
    'SmoothL1Loss': torch.nn.SmoothL1Loss(),
    'HuberLoss': torch.nn.HuberLoss(),
}

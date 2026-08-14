"""Unsupervised autoencoder anomaly detector.

Trained only on legitimate (non-fraud) transactions; the fraud score is the reconstruction
error on held-out data. This is a genuinely different modeling paradigm from the supervised
baselines — it never sees a single labeled fraud example during training. Its usefulness
rests entirely on the assumption that fraud "looks different" in reconstruction-error terms,
which does not always hold (see the Day 3 analysis writeup).
"""

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def fit_autoencoder(
    X_train_legit: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> Autoencoder:
    """Fit on legitimate transactions only. `X_train_legit` must already be preprocessed
    (scaled) — the autoencoder has no notion of raw vs. scaled features."""
    torch.manual_seed(seed)
    model = Autoencoder(input_dim=X_train_legit.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train_legit, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
    return model


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error — higher means more "anomalous"."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
    return errors.numpy()

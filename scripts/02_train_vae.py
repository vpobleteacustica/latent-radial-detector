#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scripts.vae_model import VAE, pad_or_crop, vae_loss, TARGET_FRAMES, N_MELS

"""
python -m scripts.02_train_vae
"""

DATA_ROOT = Path("data/features_logmel")
OUT_DIR   = Path("data/processed/vae_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "Batrachyla_leptopus",
    "Batrachyla_taeniata",
    "Pleurodema_thaul",
    "Calyptocephalella_gayi",
]


class LogMelDataset(Dataset):
    """
    Carga log-mel guardados como .npy, shape esperado: (N_MELS, frames).
    Aplica pad/crop para fijar frames = TARGET_FRAMES.
    """
    def __init__(self, split: str):
        self.items = []
        for sp in SPECIES:
            for p in sorted((DATA_ROOT / split / sp).glob("*.npy")):
                self.items.append((p, sp))
        if len(self.items) == 0:
            raise RuntimeError(f"No hay .npy en {DATA_ROOT}/{split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, sp = self.items[idx]
        x = np.load(p).astype(np.float32)

        # Validación mínima
        if x.ndim != 2:
            raise RuntimeError(f"Esperaba log-mel 2D (n_mels, frames). Archivo={p}, shape={x.shape}")
        if x.shape[0] != N_MELS:
            raise RuntimeError(
                f"N_MELS mismatch. Esperaba {N_MELS}, pero {p} tiene {x.shape[0]} filas."
            )

        x = pad_or_crop(x, TARGET_FRAMES)
        x = x[None, :, :]  # (1, n_mels, frames)
        return torch.from_numpy(x), sp


def main():
    device = torch.device("cpu")  # Mac CPU

    zdim = 16
    beta = 1.0
    batch_size = 32
    epochs = 50
    lr = 1e-3

    train_ds = LogMelDataset("train")
    val_ds   = LogMelDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VAE(zdim=zdim, n_mels=N_MELS, target_frames=TARGET_FRAMES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        for x, _ in tqdm(train_loader, desc=f"train ep{ep}", leave=False):
            x = x.to(device)
            xrec, mu, logvar, _ = model(x)
            loss, rec, kld = vae_loss(x, xrec, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * x.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                xrec, mu, logvar, _ = model(x)
                loss, _, _ = vae_loss(x, xrec, mu, logvar, beta=beta)
                va_loss += loss.item() * x.size(0)

        va_loss /= len(val_ds)

        print(f"epoch {ep:02d}  train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss < best_val:
            best_val = va_loss
            ckpt = {
                "zdim": zdim,
                "beta": beta,
                "target_frames": TARGET_FRAMES,
                "n_mels": N_MELS,
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, OUT_DIR / "vae_best.pt")

    print("OK. Modelo guardado en:", OUT_DIR / "vae_best.pt")


if __name__ == "__main__":
    main()
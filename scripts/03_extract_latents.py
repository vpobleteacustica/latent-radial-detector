#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

"""
Ejecutar desde la raíz del repo:
python -m scripts.03_extract_latents
"""

from scripts.vae_model import VAE, pad_or_crop, TARGET_FRAMES, N_MELS

DATA_ROOT = Path("data/features_logmel")
MODEL_PT  = Path("data/processed/vae_models/vae_best.pt")
OUT_DIR   = Path("data/processed/latents_z")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "Batrachyla_leptopus",
    "Batrachyla_taeniata",
    "Pleurodema_thaul",
    "Calyptocephalella_gayi",
]

def main():
    device = torch.device("cpu")

    if not MODEL_PT.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {MODEL_PT}")

    ckpt = torch.load(MODEL_PT, map_location=device)
    zdim = ckpt["zdim"]

    model = VAE(zdim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    for split in ["train", "val", "test"]:
        rows = []

        for sp in SPECIES:
            in_dir = DATA_ROOT / split / sp
            if not in_dir.exists():
                print(f"[WARN] No existe carpeta: {in_dir}")
                continue

            files = sorted(in_dir.glob("*.npy"))
            for p in tqdm(files, desc=f"{split}/{sp}", unit="file"):
                x = np.load(p).astype(np.float32)          # (n_mels, frames)
                x = pad_or_crop(x, TARGET_FRAMES)          # (n_mels, TARGET_FRAMES)

                x = torch.from_numpy(x[None, None, :, :])  # (1,1,n_mels,frames)
                x = x.to(device)

                with torch.no_grad():
                    mu, logvar = model.enc(x)
                    z = mu.squeeze(0).cpu().numpy()        # (zdim,) usamos mu como embedding

                # guardamos también ruta relativa para debugging (opcional)
                rel = str(p.relative_to(DATA_ROOT))

                rows.append((split, sp, p.name, rel, z))

        out_npz = OUT_DIR / f"latents_{split}.npz"

        if len(rows) == 0:
            print(f"[WARN] No se encontraron muestras para split='{split}'. No se guarda {out_npz}.")
            continue

        # OJO: NO usar key "file" aquí (colisiona con el parámetro file= de numpy)
        np.savez_compressed(
            out_npz,
            split=np.array([r[0] for r in rows]),
            label=np.array([r[1] for r in rows]),
            fname=np.array([r[2] for r in rows]),     # nombre del .npy
            relpath=np.array([r[3] for r in rows]),   # ruta relativa (opcional)
            z=np.stack([r[4] for r in rows], axis=0).astype(np.float32),
        )
        print("Guardado:", out_npz)

if __name__ == "__main__":
    main()
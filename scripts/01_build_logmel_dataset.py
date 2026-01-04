#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_build_logmel_dataset.py

Construye un dataset de log-Mel spectrograms a partir de WAVs
organizados por split y especie.

Salida:
- .npy por archivo (features)
- metadata.csv con rutas y labels

Este paso es:
- determinista
- reproducible
- explicable (NO ML aún)
"""

from pathlib import Path
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# Configuración
# =========================

DATA_ROOT = Path("data/processed/tiny_dataset_split_norm")
OUTPUT_ROOT = Path("data/features_logmel")

SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

DEVICE = "cpu"

# =========================
# Transformación log-Mel
# =========================

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
).to(DEVICE)

def wav_to_logmel(wav_path: Path) -> np.ndarray:
    wav, sr = torchaudio.load(wav_path)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(DEVICE)

    mel = mel_transform(wav)
    logmel = torch.log(mel + 1e-9)

    return logmel.squeeze(0).cpu().numpy()

# =========================
# Loop principal
# =========================

records = []

for split in ["train", "val", "test"]:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        continue

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        out_dir = OUTPUT_ROOT / split / label
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_files = list(class_dir.glob("*.wav"))

        for wav_path in tqdm(wav_files, desc=f"{split}/{label}"):
            logmel = wav_to_logmel(wav_path)

            out_path = out_dir / f"{wav_path.stem}.npy"
            np.save(out_path, logmel)

            records.append({
                "split": split,
                "label": label,
                "wav_path": str(wav_path),
                "feature_path": str(out_path),
                "frames": logmel.shape[1]
            })

# =========================
# Guardar metadata
# =========================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(OUTPUT_ROOT / "metadata.csv", index=False)

print(f"✔ Dataset log-Mel construido: {len(df)} archivos")
print(f"✔ Metadata: {OUTPUT_ROOT / 'metadata.csv'}")
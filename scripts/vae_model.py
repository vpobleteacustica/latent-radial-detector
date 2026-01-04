#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/vae_model.py

Modelo VAE (Encoder/Decoder) + utilidades para trabajar con log-Mel.
Este módulo es importado por 02_train_vae.py y 03_extract_latents.py.

Diseño:
- Mantiene la misma arquitectura que ya estabas usando.
- Centraliza constantes (TARGET_FRAMES, N_MELS) para evitar desalineaciones.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


# =========================
# Constantes del "feature space"
# =========================
# OJO: deben coincidir con cómo construyes los log-mel .npy en 01_build_logmel_dataset.py
TARGET_FRAMES: int = 256
N_MELS: int = 64


def pad_or_crop(x: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """
    Ajusta un log-mel de shape (n_mels, frames) a (n_mels, target_frames).
    """
    if x.ndim != 2:
        raise ValueError(f"pad_or_crop esperaba x 2D (n_mels, frames). Recibido shape={x.shape}")

    if x.shape[1] == target_frames:
        return x
    if x.shape[1] > target_frames:
        return x[:, :target_frames]

    pad = target_frames - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


class Encoder(nn.Module):
    def __init__(self, zdim: int, n_mels: int = N_MELS, target_frames: int = TARGET_FRAMES):
        super().__init__()
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )

        # Inferimos tamaño de salida para armar capas lineales
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_mels, self.target_frames)
            h = self.net(dummy)
            self.hshape = h.shape[1:]           # (C, H, W)
            self.hdim = int(np.prod(self.hshape))

        self.mu = nn.Linear(self.hdim, zdim)
        self.logvar = nn.Linear(self.hdim, zdim)

    def forward(self, x: torch.Tensor):
        # x: (B, 1, n_mels, frames)
        h = self.net(x).view(x.size(0), -1)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, zdim: int, hshape):
        super().__init__()
        self.hshape = hshape
        hdim = int(np.prod(hshape))
        self.fc = nn.Linear(zdim, hdim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(z.size(0), *self.hshape)
        xrec = self.net(h)
        return xrec


class VAE(nn.Module):
    def __init__(self, zdim: int, n_mels: int = N_MELS, target_frames: int = TARGET_FRAMES):
        super().__init__()
        self.zdim = zdim
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.enc = Encoder(zdim, n_mels=n_mels, target_frames=target_frames)
        self.dec = Decoder(zdim, self.enc.hshape)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        xrec = self.dec(z)
        return xrec, mu, logvar, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Devuelve un embedding determinista (mu) para usar como latente "estable".
        (Esto es lo que típicamente se usa para detección/clustering.)
        """
        mu, _ = self.enc(x)
        return mu


def vae_loss(x: torch.Tensor, xrec: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """
    Loss VAE:
      rec = MSE
      kld = KL(q(z|x)||N(0,I))
      total = rec + beta*kld
    """
    rec = torch.mean((x - xrec) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kld, rec.detach(), kld.detach()
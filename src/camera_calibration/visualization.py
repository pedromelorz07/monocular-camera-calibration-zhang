"""Figuras para inspeção das imagens e dos resíduos de reprojeção."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_sample(image_path: Path, corners: np.ndarray | None, output: Path, pattern: tuple[int, int]) -> None:
    """Salva imagem de exemplo com cantos detectados, se disponíveis."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Imagem ilegível: {image_path}")
    if corners is not None:
        cv2.drawChessboardCorners(image, pattern, corners, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Falha ao salvar figura: {output}")


def save_error_histogram(errors: list[float], output: Path) -> None:
    """Salva histograma de distâncias euclidianas por canto, em pixels."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=20, color="#2878a5", edgecolor="white")
    ax.set(title="Distribuição do erro por ponto observado", xlabel="Distância de reprojeção (px)", ylabel="Contagem")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_undistortion_comparison(original: np.ndarray, corrected: np.ndarray, output: Path) -> None:
    """Salva comparação lado a lado entre imagem original e corrigida."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)); axes[1].set_title("Corrigida")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)

"""Métricas explícitas de erro de reprojeção em pixels."""

from __future__ import annotations

import numpy as np


def reprojection_metrics(observed: list[np.ndarray], projected: list[np.ndarray]) -> dict:
    """Calcula médias/RMS por imagem e global, máximo e distâncias por observação.

    RMS por coordenadas é sqrt(mean(dx²+dy²)); média por ponto é mean(sqrt(dx²+dy²)).
    """
    if len(observed) != len(projected) or not observed:
        raise ValueError("As listas de pontos observados e projetados devem ser não vazias e ter mesmo tamanho.")
    per_view = []
    all_delta = []
    all_errors = []
    for index, (obs, proj) in enumerate(zip(observed, projected)):
        delta = np.asarray(obs, dtype=np.float64).reshape(-1, 2) - np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        if delta.shape[0] == 0 or not np.isfinite(delta).all():
            raise ValueError("Pontos vazios ou não finitos nas métricas.")
        errors = np.linalg.norm(delta, axis=1)
        per_view.append({"image_index": index, "point_count": int(errors.size),
                         "mean_px": float(errors.mean()), "rms_coordinates_px": float(np.sqrt(np.mean(delta ** 2) * 2)),
                         "max_px": float(errors.max())})
        all_delta.append(delta)
        all_errors.append(errors)
    delta_all = np.concatenate(all_delta)
    errors_all = np.concatenate(all_errors)
    return {"per_image": per_view, "global": {
        "observation_count": int(errors_all.size), "mean_per_point_px": float(errors_all.mean()),
        "rms_coordinates_px": float(np.sqrt(np.mean(delta_all ** 2) * 2)),
        "max_per_point_px": float(errors_all.max()),
        "median_per_point_px": float(np.median(errors_all)),
        "percentile_95_per_point_px": float(np.percentile(errors_all, 95)),
        "point_errors_px": errors_all.tolist()}}

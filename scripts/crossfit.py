"""Case-level halves of the held-out scanner, for cross-fitted label-free adaptation.

Adapt (AdaBN statistics, or teacher-guided adaptation) on one half of the new scanner's images, predict the other
half, then swap. Every new-scanner cell is then scored by a model that never saw its image in any form, so all 50
images of the scanner can be used for evaluation instead of only the 11 official test images.
"""
import numpy as np

def target_halves(d, holdout):
    """Images (= cases) of the held-out scanner sorted by id; half A = every other image starting with the first.
    Returns {"A": rows, "B": rows}: sorted row indices into patches.npz."""
    img = d["image_id"]; rows = np.where(d["scanner"] == holdout)[0]
    ids = np.unique(img[rows])
    return {"A": rows[np.isin(img[rows], ids[::2])], "B": rows[np.isin(img[rows], ids[1::2])]}

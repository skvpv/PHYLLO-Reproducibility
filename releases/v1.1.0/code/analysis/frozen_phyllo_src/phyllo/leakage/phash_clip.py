"""Compute frozen-detector inputs for all 7,774 images (Blueprint §4).

Detector is FROZEN — this module only computes, it never changes thresholds:
  * 64-bit pHash: ImageHash.phash(RGB, hash_size=8, highfreq_factor=4) -> 64-bit int.
  * CLIP ViT-B/32: clip.load('ViT-B/32') canonical OpenAI weights + preprocess;
    encode_image -> L2-normalized float32 embedding.

Outputs (saved by the driver): image_keys.json (ordered 'split/name'), phash_u64.npy
(uint64, one per image), clip_emb.npy (float32 [N,512], L2-normalized), and a
provenance record with package versions + SHA-256 of the downloaded CLIP weight file.

Heavy imports (torch, clip, PIL, imagehash) are done lazily so the module can be imported
on a CPU-only box for unit tests of the pure comparison code in detector.py.
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import List, Tuple
import numpy as np


def phash_to_u64(phash_obj) -> np.uint64:
    """Pack an ImageHash 8x8 boolean hash into a uint64 (row-major bit order)."""
    bits = phash_obj.hash.flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return np.uint64(v)


def compute_phash_clip(image_paths: List[str], image_keys: List[str],
                       device: str = "cuda", batch_size: int = 64):
    """Return (phash_u64[N], clip_emb[N,512] float32 L2-normalized, provenance dict).

    image_paths[i] is the on-disk path; image_keys[i] the stable 'split/name' key.
    """
    import torch
    from PIL import Image
    import imagehash
    import clip

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    phashes = np.empty(len(image_paths), dtype=np.uint64)
    embs = np.empty((len(image_paths), 512), dtype=np.float32)

    buf, idx_buf = [], []

    def flush():
        if not buf:
            return
        batch = torch.stack(buf).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs[idx_buf] = feats.cpu().numpy()
        buf.clear(); idx_buf.clear()

    for i, p in enumerate(image_paths):
        img = Image.open(p).convert("RGB")
        phashes[i] = phash_to_u64(imagehash.phash(img, hash_size=8, highfreq_factor=4))
        buf.append(preprocess(img)); idx_buf.append(i)
        if len(buf) >= batch_size:
            flush()
    flush()

    prov = {
        "detector": "FROZEN pHash(hash_size=8,highfreq_factor=4) UNION CLIP ViT-B/32",
        "n_images": len(image_paths),
        "clip_weight_sha256": _clip_weight_sha256(),
        "package_versions": _versions(),
    }
    return phashes, embs, prov


def _clip_weight_sha256() -> str:
    """SHA-256 of the cached CLIP ViT-B/32 weight file (recorded, model unchanged)."""
    cache = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
    if not os.path.exists(cache):
        return "NOT_FOUND (record on first download)"
    h = hashlib.sha256()
    with open(cache, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _versions() -> dict:
    out = {}
    for mod in ("torch", "clip", "imagehash", "PIL", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "absent"
    return out

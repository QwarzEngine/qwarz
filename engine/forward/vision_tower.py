"""BF16 vision tower from the pinned EXL3 artifact.

Image rows stay out of the text table. The token row keeps the artifact's
vision start and end ids around dynamic ids at or above 1000000000.
"""
from __future__ import annotations

import io
from pathlib import Path

from engine.forward.vision import MM_TOKEN_BASE, text_table_ids

VISION_START = 248053
VISION_END = 248054


def embed_image(model_dir, image):
    from exllamav3 import Tokenizer
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    import torch

    from qwasar_runtime.vision import max_pixels_setting

    directory = Path(model_dir)
    config = Config.from_directory(directory)
    vision_pp = getattr(config, "vision_pp", None)
    if vision_pp is not None and hasattr(vision_pp, "max_pixels"):
        vision_pp.max_pixels = max_pixels_setting()
    tower = Model.from_config(config, component="vision")
    tower.load(device="cuda:0")
    try:
        embedding = tower.get_image_embeddings(Tokenizer.from_config(config), image)
    finally:
        tower.unload()
        torch.cuda.empty_cache()
    tokens = list(embedding.token_list)
    rows = embedding.embeddings
    dynamic = [token for token in tokens if token >= MM_TOKEN_BASE]
    if tokens[0] != VISION_START or tokens[-1] != VISION_END:
        raise RuntimeError("vision token row is missing the artifact start or end id")
    if len(dynamic) != rows.shape[0]:
        raise RuntimeError("vision rows do not match the dynamic image ids")
    return {
        "tokens": tokens,
        "rows": rows,
        "dynamic_ids": dynamic,
        "grid_thw": embedding.grid_thw,
        "merge_size": embedding.mrope_merge_size,
        "first_index": embedding.first_index,
        "last_index": embedding.last_index,
    }


def mix_rows(table, ids, rows, dynamic_ids):
    """Text rows come from the table. Image ids are replaced and never gathered."""
    import torch

    from engine.forward.embed import gather

    lookup = {token: index for index, token in enumerate(dynamic_ids)}
    plain = text_table_ids(ids)
    gathered = gather(table, torch.tensor([plain], dtype=torch.long), torch.float32).cuda().contiguous()
    placed = rows.float().cuda()
    for position, token in enumerate(ids):
        if token >= MM_TOKEN_BASE:
            gathered[0, position].copy_(placed[lookup[token]])
    return gathered


def mrope_freqs(ids, first_index, last_index, grid_thw, merge_size):
    """Per-token RoPE table for one image span, the same one ExLlama stores on the job."""
    import torch
    from types import SimpleNamespace

    from engine.forward.attention import rope

    # gen_mrope_pos_ids writes a CPU table. The frequency vector lives on CPU
    # until the matmul, then the attention kernel reads it on the device.
    embedding = SimpleNamespace(
        first_index=first_index,
        last_index=last_index,
        grid_thw=grid_thw,
        mrope_merge_size=merge_size,
    )
    freqs, _offset = rope("cpu").get_mrope_freqs(
        torch.tensor([list(ids)], dtype=torch.long),
        [embedding],
        len(ids),
    )
    return freqs.cuda().contiguous()


def mrope_table(ids, images, length):
    """One MRoPE table for every image in the prompt, long enough for the decode."""
    import torch
    from types import SimpleNamespace

    from engine.forward.attention import rope

    embeddings = [
        SimpleNamespace(
            first_index=image["first_index"],
            last_index=image["last_index"],
            grid_thw=image["grid_thw"],
            mrope_merge_size=image["merge_size"],
        )
        for image in images
    ]
    width = max(int(length), len(ids))
    freqs, _offset = rope("cpu").get_mrope_freqs(
        torch.tensor([list(ids)], dtype=torch.long),
        embeddings,
        width,
    )
    return freqs.cuda().contiguous()


def png_bytes(width, height, rgb=(20, 40, 60)):
    from PIL import Image

    image = Image.new("RGB", (width, height), rgb)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

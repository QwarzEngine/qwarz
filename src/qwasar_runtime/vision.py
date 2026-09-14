"""Native image input for Qwen3.8-27B.

The pinned EXL3 artifact already ships the BF16 vision tower
(``model.visual.*``). This module loads that component next to the text
generator, turns request images into ExLlamaV3 ``MMEmbedding`` objects and
keeps them in a small LRU so repeated turns reuse the same dynamic token IDs
(which is what keeps prefix-cache reuse working across a session).

Dynamic image token IDs start at ``MM_TOKEN_BASE`` and are process-local, so
persisted tapes canonicalize them with :func:`canonical_tape`.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import os
import struct


MM_TOKEN_BASE = 1_000_000_000  # exllamav3.tokenizer.mm_embedding.FIRST_MM_EMBEDDING_INDEX
DEFAULT_MAX_PIXELS = 2_359_296  # ~2304 image tokens; a 1920x1080 frame is not downscaled
MIN_MAX_PIXELS = 65_536
DEFAULT_CACHE_ENTRIES = 32
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PATCH_PIXELS = 32  # patch_size 16 * spatial_merge_size 2
IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")


def max_pixels_setting():
    value = int(os.environ.get("QWASAR_MAX_IMAGE_PIXELS", DEFAULT_MAX_PIXELS))
    if value < MIN_MAX_PIXELS:
        raise ValueError(f"QWASAR_MAX_IMAGE_PIXELS must be at least {MIN_MAX_PIXELS}")
    return value


def cache_entries_setting():
    value = int(os.environ.get("QWASAR_IMAGE_CACHE", DEFAULT_CACHE_ENTRIES))
    if value < 1:
        raise ValueError("QWASAR_IMAGE_CACHE must be at least 1")
    return value


def canonical_tape(tokens):
    """Replaces process-local image token IDs with -1 so persisted tapes compare
    equal across reloads and cache evictions."""
    return [token if token < MM_TOKEN_BASE else -1 for token in tokens]


@dataclass
class ImageEmbedding:
    sha256: str
    tokens: list[int]
    width: int
    height: int
    handle: object = None  # exllamav3 MMEmbedding; None for the fake runtime


def decode_image_data(sha256, media_type, data):
    """Decodes the base64 payload the supervisor attached and checks it against
    the hash the message references."""
    if media_type not in IMAGE_MEDIA_TYPES:
        raise ValueError("unsupported image media type")
    if not isinstance(data, str):
        raise ValueError(f"image {sha256} has no inline data")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("image data is not valid base64") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must contain 1..{MAX_IMAGE_BYTES} bytes")
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"image data does not match sha256 {sha256}")
    return raw


class EmbeddingCache:
    def __init__(self, entries):
        self.entries = entries
        self._items: OrderedDict[str, ImageEmbedding] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, sha256):
        item = self._items.get(sha256)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(sha256)
        self.hits += 1
        return item

    def put(self, item):
        self._items[item.sha256] = item
        self._items.move_to_end(item.sha256)
        while len(self._items) > self.entries:
            self._items.popitem(last=False)
        return item

    def __len__(self):
        return len(self._items)


def png_size(raw):
    """Width/height from a PNG header without any imaging library (fake runtime)."""
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError("fake runtime only decodes PNG images")
    width, height = struct.unpack(">II", raw[16:24])
    if not width or not height:
        raise ValueError("image has empty dimensions")
    return width, height


def fake_tokens(sha256, width, height, max_pixels, start_token, end_token):
    """Deterministic stand-in for MMEmbedding.token_list: one token per 32x32
    block after clamping to max_pixels, in a sha-derived dynamic ID range."""
    pixels = width * height
    scale = min(1.0, (max_pixels / pixels) ** 0.5)
    columns = max(1, round(width * scale / PATCH_PIXELS))
    rows = max(1, round(height * scale / PATCH_PIXELS))
    base = MM_TOKEN_BASE + (int(sha256[:8], 16) % 4096) * 65536
    return [start_token] + list(range(base, base + rows * columns)) + [end_token]


class VisionRuntime:
    """Owns the loaded vision component and the embedding LRU."""

    def __init__(self, generator, tokenizer, max_pixels=None, cache_entries=None, device="cuda:0"):
        from exllamav3 import Model

        config = generator.model.config
        if not getattr(config, "vision", None) or "vision" not in config.model_classes:
            raise ValueError("artifact does not declare a vision component")
        self.max_pixels = max_pixels_setting() if max_pixels is None else max_pixels
        config.vision_pp.max_pixels = self.max_pixels
        self.tokenizer = tokenizer
        self.model = Model.from_config(config, component="vision")
        # Single-device load: the autosplit path derives its per-process memory
        # fraction from *free* VRAM, which is wrong once the text model already
        # owns ~28 GB in this process.
        self.model.load(device=device)
        self.cache = EmbeddingCache(cache_entries_setting() if cache_entries is None else cache_entries)

    def embed(self, sha256, media_type, data):
        cached = self.cache.get(sha256)
        if cached is not None:
            return cached
        raw = decode_image_data(sha256, media_type, data)
        from PIL import Image

        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 64_000_000
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as error:  # PIL raises many unrelated classes
            raise ValueError(f"image {sha256} could not be decoded: {error}") from error
        finally:
            Image.MAX_IMAGE_PIXELS = previous
        if getattr(image, "is_animated", False):
            image.seek(0)
        width, height = image.size
        embedding = self.model.get_image_embeddings(self.tokenizer, image)
        return self.cache.put(ImageEmbedding(sha256, list(embedding.token_list), width, height, embedding))

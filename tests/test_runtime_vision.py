"""Image input through the fake runtime: parsing, rendering, engine and tape canonicalization."""
import base64
import hashlib
import struct
import threading
import unittest
import zlib

from qwasar_runtime import vision
from qwasar_runtime.engine import Engine, FakeBackend
from qwasar_runtime.parsing import normalize_parts, validate_messages
from qwasar_runtime.rendering import render


def png_bytes(width, height, color=(200, 30, 30)):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def image_request(raw, text="what is this?", media_type="image/png", extra_images=None):
    sha256 = hashlib.sha256(raw).hexdigest()
    parts = [{"type": "text", "text": text}, {"type": "image", "sha256": sha256, "media_type": media_type}]
    images = {sha256: {"media_type": media_type, "data": base64.b64encode(raw).decode()}}
    images.update(extra_images or {})
    return sha256, {"messages": [{"role": "user", "content": parts}], "images": images}


class PartsTests(unittest.TestCase):
    def test_text_only_parts_collapse_to_string_and_adjacent_text_merges(self):
        self.assertEqual(normalize_parts([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "user"), "ab")
        sha = "0" * 64
        parts = normalize_parts([{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                                 {"type": "image", "sha256": sha, "media_type": "image/png"}], "user")
        self.assertEqual(parts, [{"type": "text", "text": "ab"}, {"type": "image", "sha256": sha, "media_type": "image/png"}])

    def test_images_rejected_outside_user_and_with_bad_metadata(self):
        image = {"type": "image", "sha256": "0" * 64, "media_type": "image/png"}
        with self.assertRaises(ValueError):
            normalize_parts([image], "assistant")
        with self.assertRaises(ValueError):
            normalize_parts([{**image, "sha256": "ABC"}], "user")
        with self.assertRaises(ValueError):
            normalize_parts([{**image, "media_type": "image/bmp"}], "user")
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}], [])


class RenderTests(unittest.TestCase):
    def test_image_tokens_are_spliced_literally_between_text(self):
        tokenizer = FakeBackend().tokenizer
        sha = "a" * 64
        tokens = [1000005, vision.MM_TOKEN_BASE, vision.MM_TOKEN_BASE + 1, 1000006]
        messages = [{"role": "user", "content": [{"type": "text", "text": "before "},
                     {"type": "image", "sha256": sha, "media_type": "image/png"}, {"type": "text", "text": " after"}]}]
        rendered, _, _ = render(tokenizer, messages, [], "off", "auto", [], {sha: tokens})
        start = rendered.index(1000005)
        self.assertEqual(rendered[start:start + 4], tokens)
        self.assertEqual(rendered[start - len("before "):start], [ord(c) for c in "before "])
        self.assertEqual(rendered[start + 4:start + 4 + len(" after")], [ord(c) for c in " after"])
        with self.assertRaises(ValueError):
            render(tokenizer, messages, [], "off", "auto", [], {})


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.engine = Engine(self.backend, context_size=8192)

    def run_request(self, request, parent=None):
        events = list(self.engine.generate("resp_vision", request, parent, threading.Event()))
        self.assertEqual(events[-1]["type"], "terminal")
        return events

    def test_image_costs_tokens_and_snapshot_tape_is_canonical(self):
        raw = png_bytes(64, 96)
        sha, request = image_request(raw)
        text_only = self.engine.prepare({"messages": [{"role": "user", "content": "what is this?"}]}, None)
        prompt = self.engine.prepare(request, None)
        self.assertEqual(len(prompt.tokens) - len(text_only.tokens), 2 * 3 + 2)
        self.assertEqual(len(prompt.request["_embeddings"]), 0)  # fake handles are None
        terminal = self.run_request(image_request(raw)[1])[-1]
        self.assertEqual(terminal["status"], "completed")
        tape = terminal["snapshot"]["tape"]
        self.assertEqual(tape.count(-1), 6)
        self.assertTrue(all(token < vision.MM_TOKEN_BASE for token in tape))
        self.assertEqual(terminal["snapshot"]["messages"][0]["content"][1]["sha256"], sha)
        self.assertEqual(terminal["usage"]["prompt_tokens"], len(prompt.tokens))

    def test_followup_with_same_image_keeps_parent_prefix(self):
        raw = png_bytes(64, 64)
        sha, request = image_request(raw)
        first = self.run_request(request)[-1]
        snapshot = first["snapshot"]
        followup = {"messages": snapshot["messages"] + [{"role": "user", "content": "and now?"}], "images": request["images"]}
        prompt = self.engine.prepare(followup, snapshot)
        self.assertEqual(vision.canonical_tape(prompt.tokens[:len(snapshot["tape"])]), snapshot["tape"])
        self.assertEqual(len(prompt.segments), 1)
        self.assertEqual(self.backend.images.hits, 1)
        # A fresh process gets new dynamic IDs; segments still apply and nothing is ambiguous.
        fresh = Engine(FakeBackend(), context_size=8192)
        prompt = fresh.prepare(followup, snapshot)
        self.assertEqual(len(prompt.segments), 1)

    def test_missing_or_corrupt_image_data_is_a_request_error(self):
        raw = png_bytes(32, 32)
        sha, request = image_request(raw)
        del request["images"][sha]
        terminal = self.run_request(request)[-1]
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"]["code"], "unknown_image")
        sha, request = image_request(raw)
        request["images"][sha]["data"] = base64.b64encode(png_bytes(32, 33)).decode()
        terminal = self.run_request(request)[-1]
        self.assertIn("does not match sha256", terminal["error"]["message"])
        self.assertEqual(self.backend.enqueued, 0)

    def test_too_many_images_rejected_before_enqueue(self):
        raws = [png_bytes(32, 32, (i, i, i)) for i in range(vision.MAX_IMAGES + 1)]
        parts, images = [], {}
        for raw in raws:
            sha = hashlib.sha256(raw).hexdigest()
            parts.append({"type": "image", "sha256": sha, "media_type": "image/png"})
            images[sha] = {"media_type": "image/png", "data": base64.b64encode(raw).decode()}
        terminal = self.run_request({"messages": [{"role": "user", "content": parts}], "images": images})[-1]
        self.assertEqual(terminal["status"], "failed")
        self.assertIn("at most", terminal["error"]["message"])
        self.assertEqual(self.backend.enqueued, 0)

    def test_image_budget_counts_against_context(self):
        raw = png_bytes(1920, 1080)
        sha, request = image_request(raw)
        request["max_tokens"] = 8192 - 100
        terminal = self.run_request(request)[-1]
        self.assertEqual(terminal["error"]["code"], "context_length_exceeded")


class CacheTests(unittest.TestCase):
    def test_lru_evicts_oldest_and_canonical_tape_masks_dynamic_ids(self):
        cache = vision.EmbeddingCache(2)
        for name in "abc":
            cache.put(vision.ImageEmbedding(name, [], 1, 1))
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        self.assertEqual(vision.canonical_tape([5, vision.MM_TOKEN_BASE + 7, 9]), [5, -1, 9])

    def test_fake_tokens_scale_with_pixels_and_cap(self):
        small = vision.fake_tokens("0" * 64, 64, 64, vision.DEFAULT_MAX_PIXELS, 1, 2)
        self.assertEqual(len(small), 2 + 4)
        huge = vision.fake_tokens("0" * 64, 8000, 8000, vision.DEFAULT_MAX_PIXELS, 1, 2)
        self.assertLessEqual(len(huge) - 2, vision.DEFAULT_MAX_PIXELS // (32 * 32) + 64)
        with self.assertRaises(ValueError):
            vision.png_size(b"not a png")


if __name__ == "__main__":
    unittest.main()

"""
darkclaw/core/vision.py

Image understanding via moondream on the memory node.

Images cost zero P100 VRAM: moondream (1.8B) runs CPU-side on the
Lenovo memory node. Ingestion is asynchronous background work, so
CPU-speed description (~30-60s) is invisible to the user.

The describer returns plain text; doc_store feeds it through the
normal chunk → memory pipeline, which makes uploaded images fully
queryable by every agent like any other document.
"""
import base64
import json
import os
import urllib.request

MEMORY_NODE_URL = os.environ.get("DARKCLAW_MEMORY_NODE_URL",
                                 "http://192.168.1.130:11434").rstrip("/")
VISION_MODEL    = os.environ.get("DARKCLAW_VISION_MODEL", "moondream:latest")
VISION_TIMEOUT  = int(os.environ.get("DARKCLAW_VISION_TIMEOUT", "180"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

_PROMPT = (
    "Describe this image in detail. Include: what it shows, any visible "
    "text (transcribe it exactly), numbers, labels, UI elements, objects, "
    "and their arrangement. Be thorough and factual."
)


def describe_image(raw_bytes: bytes, prompt: str = _PROMPT) -> str:
    """
    Send an image to moondream on the memory node, return its description.
    Raises on failure — callers decide how to degrade.
    """
    body = json.dumps({
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(raw_bytes).decode()],
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        f"{MEMORY_NODE_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=VISION_TIMEOUT) as r:
        out = json.loads(r.read().decode()).get("response", "").strip()
    if not out:
        raise ValueError("moondream returned an empty description")
    return out

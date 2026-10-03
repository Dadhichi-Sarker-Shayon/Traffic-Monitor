"""Qwen2-VL backend for the adjudicator - in-process or as a worker process.

OpenPSG needs the old torch 1.13 / mmcv 1.x stack, which cannot host a modern
`transformers`. So the model can also run in a separate interpreter:

    python traffic_watch/vlm_backend.py --model Qwen/Qwen2-VL-2B-Instruct

The worker reads one JSON request per stdin line, {"image": path, "q": text},
and writes one JSON reply per stdout line, {"answer": text} or {"error": text}.
This file is therefore self-contained (no package imports) and is also what
the in-process path uses.
"""

from __future__ import annotations

import json
import sys


class QwenVL:
    """Yes/no visual question answering with Qwen2-VL (4-bit when on CUDA)."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cuda", max_pixels: int = 400 * 28 * 28,
                 adapter: str = None):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        if "2.5" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2VLForConditionalGeneration as Cls
        kw = {}
        if self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4")
                kw["device_map"] = {"": 0}
            except Exception:
                kw["torch_dtype"] = torch.float16
        self.model = Cls.from_pretrained(model_id, **kw)
        if adapter:                      # LoRA adapter from notebooks/02_vlm_crash_qlora.ipynb
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if self.device == "cpu" or "device_map" not in kw:
            self.model.to(self.device)
        self.model.eval()

    def ask(self, image_rgb, question: str, max_new_tokens: int = 4) -> str:
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[pil], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    def __call__(self, image_rgb, prompt=None):
        """Same call shape as a transformers image-to-text pipeline."""
        return [{"generated_text": self.ask(image_rgb, prompt or "")}]


class WorkerPipe:
    """Talks to a `vlm_backend.py` worker running under another interpreter."""

    def __init__(self, python_exe: str, model_id: str, device: str = "cuda",
                 timeout_s: float = 180.0, adapter: str = None):
        import os
        import subprocess
        import tempfile

        self._tmp = tempfile.mkdtemp(prefix="vlm_")
        self.timeout_s = timeout_s
        self._n = 0
        self._errpath = os.path.join(self._tmp, "stderr.txt")
        self._err = open(self._errpath, "w")
        self.proc = subprocess.Popen(
            [python_exe, os.path.abspath(__file__), "--model", model_id, "--device", device]
            + (["--adapter", adapter] if adapter else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            text=True, bufsize=1)
        ready = self._readline(self.timeout_s * 3)
        if ready is None or json.loads(ready).get("ready") is not True:
            self.close()
            self._err.flush()
            tail = open(self._errpath, errors="ignore").read().strip().splitlines()[-3:]
            raise RuntimeError("vlm worker failed to start: " + " | ".join(tail)[-300:])

    def _readline(self, timeout):
        import queue
        import threading

        if not hasattr(self, "_q"):
            self._q = queue.Queue()

            def pump():
                for line in self.proc.stdout:
                    self._q.put(line)
                self._q.put(None)
            threading.Thread(target=pump, daemon=True).start()
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __call__(self, image_rgb, prompt=None):
        import os

        import cv2

        self._n += 1
        path = os.path.join(self._tmp, f"f{self._n % 4}.jpg")
        cv2.imwrite(path, image_rgb[:, :, ::-1])
        self.proc.stdin.write(json.dumps({"image": path, "q": prompt or ""}) + "\n")
        self.proc.stdin.flush()
        line = self._readline(self.timeout_s)
        if not line:
            raise RuntimeError("vlm worker timed out or died")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return [{"generated_text": reply["answer"]}]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def serve(model_id: str, device: str, adapter: str = None) -> None:
    import cv2

    vl = QwenVL(model_id, device, adapter=adapter)
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            img = cv2.imread(req["image"])
            rgb = img[:, :, ::-1].copy()
            reply = {"answer": vl.ask(rgb, req["q"])}
        except Exception as e:  # keep serving whatever one request does
            reply = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    serve(a.model, a.device, a.adapter)

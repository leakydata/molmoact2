"""MolmoAct2-SO100_101 inference server.

Mirrors `host_server_yam.py` but for the single-arm SO-100/101 checkpoint:

  * 2 cameras: a third-person `scene_cam` and a `wrist_cam` (the checkpoint
    card says camera order does not matter; we still keep [scene, wrist])
  * raw robot state is shape (6,) in the *LeRobot v2.1* degree convention:
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
  * norm_tag = "so100_so101_molmoact2"

One request = one arm. To drive two arms, the client calls /act once per arm.

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", ...} plus the
                        checkpoint's state q01/q99 so clients can sanity-check
                        their joint-frame conversion before moving the arm
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "scene_cam":   ndarray(H, W, 3) uint8 RGB,
              "wrist_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(6,) float32,
              "timestamp":   float (optional),
              "num_steps":   int   (optional, default 10),
              "enable_cuda_graph": bool (optional),
              "seed":        int   (optional, flow-matching noise seed),
            }
        response body (json_numpy):
            {"actions": ndarray(N, 6) float32, "dt_ms": float}

Run:

    uv run python examples/so101/host_server_so101.py --host 0.0.0.0 --port 8101
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
# Must be called before any json.dumps/loads we rely on.
json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.so101.server")


REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
STATE_DIM = 6
NUM_CAMERAS = 2
DEFAULT_NUM_STEPS = 10


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Same idempotent patches as the DROID/YAM servers. Needles that no
    longer match a newer `modeling_molmoact2.py` warn rather than fail.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


def _enable_multi_graph_cache(model: Any, capacity: int) -> None:
    """Keep up to `capacity` action-expert CUDA graphs instead of one.

    Upstream `ActionCudaGraphManager` caches a single graph keyed on the
    context shape, which includes the prompt's token count. Two arms with
    different-length prompts therefore re-capture (~2.5 s) on every request.
    Replace `run_action_flow` per instance with a small LRU; the capture helpers
    are taken from the remote-code module so the logic stays upstream's.
    """
    manager = getattr(getattr(model, "model", None), "action_cuda_graph_manager", None)
    if manager is None or capacity <= 1:
        return
    mod = sys.modules[type(manager).__module__]
    graphs: collections.OrderedDict = collections.OrderedDict()

    def run_action_flow(inputs: Any, steps: int, run_loop: Any) -> torch.Tensor:
        key = mod._cuda_graph_key(inputs, steps)
        cache = graphs.get(key)
        if cache is None:
            while len(graphs) >= capacity:
                graphs.popitem(last=False)
            static_inputs = mod._clone_static_inputs(inputs)
            graph, output = mod._capture_cuda_graph(
                lambda: run_loop(static_inputs, steps),
                inputs.trajectory.device,
                after_warmup=lambda: static_inputs.trajectory.copy_(inputs.trajectory),
            )
            cache = mod._ActionFlowCudaGraph(
                key=key, graph=graph, static_inputs=static_inputs, output=output
            )
            graphs[key] = cache
            log.info("captured action CUDA graph %d/%d", len(graphs), capacity)
        else:
            graphs.move_to_end(key)
            mod._copy_inputs_(cache.static_inputs, inputs)
        manager.action_flow_graph = cache
        cache.graph.replay()
        return cache.output.clone()

    manager.run_action_flow = run_action_flow


def _load_state_stats(local_dir: str) -> dict[str, Any]:
    """Pull the per-joint state quantiles out of `norm_stats.json`."""
    try:
        with open(os.path.join(local_dir, "norm_stats.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)["metadata_by_tag"][NORM_TAG]
    except (OSError, KeyError, ValueError):
        log.warning("could not read state stats for %s from norm_stats.json", NORM_TAG)
        return {}
    stats = meta.get("state_stats", {})
    return {
        "joint_names": stats.get("names"),
        "state_q01": stats.get("q01"),
        "state_q99": stats.get("q99"),
        "state_min": stats.get("min"),
        "state_max": stats.get("max"),
        "action_horizon": meta.get("action_horizon"),
    }


class Policy:
    """Holds the loaded model + processor and serializes inference calls."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
        cuda_graph_cache: int = 2,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        # `predict_action` reads `norm_stats.json` from `config._name_or_path`.
        # Always resolve to the local snapshot dir so that lookup works.
        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved snapshot dir: %s", local_dir)
        self.state_stats = _load_state_stats(local_dir)

        _patch_modeling_for_bf16(local_dir)

        log.info("Loading processor")
        # `tokenizer_config.json` ships `extra_special_tokens` as a list, which
        # transformers >=4.46 rejects. The model code only uses these via
        # `convert_tokens_to_ids`, so an empty dict is safe.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        # Upstream `_move_inputs_to_device` only moves tensors; it does not
        # cast floats to the model dtype. With bf16 weights the processor's
        # fp32 `pixel_values` then trips `mat1 and mat2 must have the same
        # dtype`. Replace the bound method per-instance.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        _enable_multi_graph_cache(self.model, cuda_graph_cache)
        # CUDA-graph capture in the action expert is not safe under concurrent
        # calls; with two arms the client issues back-to-back requests.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(
        self,
        scene_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        images = [_to_pil(scene_cam), _to_pil(wrist_cam)]
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {state_f32.shape}"
            )

        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
                generator=None if seed is None
                else torch.Generator(device=self.device).manual_seed(int(seed)),
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-SO100_101 server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": REPO_ID,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.model.dtype),
                "num_cameras": NUM_CAMERAS,
                "state_dim": STATE_DIM,
                **policy.state_stats,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            scene_cam = payload["scene_cam"]
            wrist_cam = payload["wrist_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        seed = payload.get("seed")
        enable_cuda_graph = bool(
            payload.get("enable_cuda_graph", policy.default_cuda_graph)
        )

        t0 = time.perf_counter()
        try:
            actions = policy.predict(
                scene_cam=scene_cam,
                wrist_cam=wrist_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
                enable_cuda_graph=enable_cuda_graph,
                seed=None if seed is None else int(seed),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with dummy frames (cuda_graph=%s) ...",
             policy.default_cuda_graph)
    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        policy.predict(
            scene_cam=dummy_img,
            wrist_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            enable_cuda_graph=policy.default_cuda_graph,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-SO100_101 inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8101, help="bind port (default: 8101)")
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype (default: bfloat16, <16 GB; fp32 needs ~24-26 GB)",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph capture for action expert (faster but ~2 GB more VRAM)",
    )
    p.add_argument(
        "--cuda-graph-cache",
        type=int,
        default=2,
        help="action CUDA graphs kept at once; one per distinct prompt length "
        "(default: 2, enough for two arms)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    policy = Policy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
        cuda_graph_cache=args.cuda_graph_cache,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

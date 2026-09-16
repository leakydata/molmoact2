"""Async inference + temporal-ensembling runtime for one or more SO-101 arms.

    InferenceProducer ──POST /act──► host_server_so101.py
          │  (round-robins over arms; one request per arm per cycle)
          ▼
    ChunkRingBuffer[arm] ◄── ExecutionConsumer[arm] ──set_target──► FollowerArm

The producer captures (scene, wrist, joint state) for an arm, converts the
state into the model's joint convention, asks the server for an action chunk
(30 steps @ 30 fps), converts it back to the arm's convention and stores it,
tagged with the observation time.

Each arm's consumer ticks at `exec_hz`, finds every stored chunk whose window
covers "now", and blends their actions with weights exp(-ensemble_m * age)
(ACT-style temporal ensembling adapted to async chunks), then applies the
per-tick motion cap before handing the target to the arm's serial thread.

Joint conventions (LeRobot backward-compat doc):
    state_model = signs * state_arm + offsets
    action_arm  = (action_model - offsets) * signs

Adapted from https://github.com/irenegracekp/molmoact2-so101.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import json_numpy
import numpy as np
import requests

from so101_hardware import JOINT_COUNT, FollowerArm

ACTION_FPS = 30.0


class PolicyClient:
    """Hand-rolled HTTP client for `host_server_so101.py`."""

    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def health(self) -> dict:
        r = self._session.get(f"{self.url}/act", timeout=5.0)
        r.raise_for_status()
        return r.json()

    def act(self, scene_rgb: np.ndarray, wrist_rgb: np.ndarray, instruction: str,
            state: np.ndarray, num_steps: int, seed: int | None = None) -> tuple[np.ndarray, float]:
        body = json_numpy.dumps({
            **({} if seed is None else {"seed": int(seed)}),
            "scene_cam": np.ascontiguousarray(scene_rgb, dtype=np.uint8),
            "wrist_cam": np.ascontiguousarray(wrist_rgb, dtype=np.uint8),
            "instruction": instruction,
            "state": np.asarray(state, dtype=np.float32),
            "num_steps": int(num_steps),
            "timestamp": time.time(),
        })
        r = self._session.post(f"{self.url}/act", data=body, timeout=self.timeout_s,
                               headers={"Content-Type": "application/json"})
        out = json_numpy.loads(r.text)
        if r.status_code != 200:
            raise RuntimeError(f"server {r.status_code}: {out.get('error', r.text[:200])}")
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] != JOINT_COUNT:
            raise RuntimeError(f"unexpected action shape {actions.shape}, expected (T, {JOINT_COUNT})")
        return actions, float(out.get("dt_ms", 0.0))


def clip_step(target: np.ndarray, current: np.ndarray, max_step_deg: float) -> np.ndarray:
    """Scale the whole delta so no joint moves more than `max_step_deg` in one tick."""
    delta = target - current
    biggest = float(np.max(np.abs(delta)))
    if biggest <= max_step_deg or biggest == 0.0:
        return target
    return current + delta * (max_step_deg / biggest)


class ChunkRingBuffer:
    """Thread-safe ring of recent (chunk_in_arm_frame, t_start, chunk_id)."""

    def __init__(self, capacity: int = 8):
        self._lock = threading.Lock()
        self._entries: collections.deque = collections.deque(maxlen=capacity)
        self._next_id = 1

    def add(self, chunk: np.ndarray, t_start: float) -> int:
        with self._lock:
            chunk_id = self._next_id
            self._next_id += 1
            self._entries.append((chunk, t_start, chunk_id))
            return chunk_id

    def clear(self) -> None:
        """Drop queued plans — they were computed for a pose we have left."""
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> list:
        with self._lock:
            return list(self._entries)


@dataclass
class ArmRuntime:
    """Everything the runtime needs to know about one arm."""
    name: str
    follower: FollowerArm
    scene_cam: object
    wrist_cam: object
    prompt: str
    signs: np.ndarray
    offsets: np.ndarray
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_step_deg: float = 5.0
    # If the first action of a chunk is further than this from the arm's
    # current pose, the chunk is dropped. A wrong joint convention shows up
    # exactly like this, so it is the main guard against slamming the arm.
    max_chunk_jump_deg: float = 30.0
    scene_only: bool = False
    # If set, the prompt is re-read from this file whenever it changes, so a
    # task can be staged ("pick up X" -> "put X in Y") without reconnecting.
    prompt_file: str | None = None
    # While time.monotonic() < hold_until the consumer stops issuing targets,
    # so a supervisor can re-home the arm without fighting the policy.
    hold_until: float = 0.0
    # Largest motion the newest plan asks for, and when it landed; a supervisor
    # uses this to spot the policy stalling in a pose it does not like.
    last_plan_peak: float = 0.0
    last_plan_t: float = 0.0
    recent_peaks: collections.deque = field(default_factory=lambda: collections.deque(maxlen=20))
    _prompt_mtime: float = 0.0
    status: str = "starting"          # one-line summary for the preview
    ring: ChunkRingBuffer = field(default_factory=ChunkRingBuffer)

    def to_model_frame(self, state_arm: np.ndarray) -> np.ndarray:
        return (self.signs * state_arm + self.offsets).astype(np.float32)

    def to_arm_frame(self, actions_model: np.ndarray) -> np.ndarray:
        return ((actions_model - self.offsets) * self.signs).astype(np.float32)


@dataclass
class RuntimeConfig:
    exec_hz: float = 30.0
    num_steps: int = 10
    ensemble_m: float = 0.5
    smooth_alpha: float = 1.0
    actions_per_chunk: int | None = None
    warmup_predictions: int = 1
    max_latency_skip_s: float = 0.3
    # "ensemble": blend all live chunks (smooth, but rare reaching plans get
    # averaged away). "commit": follow one chunk for commit_steps, then jump
    # to the newest.
    execution_mode: str = "ensemble"
    commit_steps: int = 15
    # The policy is stochastic: in a hard scene most samples are "approach with
    # open jaws" and only a few actually close the gripper. >1 draws that many
    # plans per cycle and keeps the one that commits most to a grasp. This picks
    # among the policy's own proposals; it does not author any motion.
    best_of: int = 1
    save_frames_dir: str | None = None
    dry_run: bool = False


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:6.1f}" for x in np.asarray(v, dtype=float)) + "]"


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _grip_travel(actions_model: np.ndarray) -> float:
    """How much a plan opens/closes the gripper. ~0 means 'approach, never grasp'."""
    g = np.asarray(actions_model, dtype=np.float32)[:, 5]
    return float(g.max() - g.min())


class _InferenceProducer(threading.Thread):
    def __init__(self, client: PolicyClient, arms: list[ArmRuntime], cfg: RuntimeConfig):
        super().__init__(daemon=True, name="InferenceProducer")
        self.client = client
        self.arms = arms
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._preds = {a.name: 0 for a in arms}
        self._last_warn = {a.name: 0.0 for a in arms}

    def stop(self) -> None:
        self._stop_event.set()

    def _step(self, arm: ArmRuntime) -> None:
        cfg = self.cfg
        if arm.prompt_file:
            try:
                mtime = os.path.getmtime(arm.prompt_file)
                if mtime != arm._prompt_mtime:
                    text = open(arm.prompt_file, encoding="utf-8").read().strip()
                    arm._prompt_mtime = mtime
                    if text and text != arm.prompt:
                        arm.prompt = text
                        print(f"[{arm.name}] prompt -> {text!r}")
            except OSError:
                pass
        scene = arm.scene_cam.read()
        wrist = scene if arm.scene_only else arm.wrist_cam.read()
        if scene is None or wrist is None:
            if time.monotonic() - self._last_warn[arm.name] > 5.0:
                print(f"[{arm.name}] waiting for camera frames")
                self._last_warn[arm.name] = time.monotonic()
            time.sleep(0.05)
            return
        state_arm = arm.follower.get_state()
        state_model = arm.to_model_frame(state_arm)

        if cfg.save_frames_dir:
            ts = int(time.time() * 1000)
            cv2.imwrite(os.path.join(cfg.save_frames_dir, f"{ts:013d}_{arm.name}_scene.jpg"), scene)
            cv2.imwrite(os.path.join(cfg.save_frames_dir, f"{ts:013d}_{arm.name}_wrist.jpg"), wrist)

        t_obs = time.monotonic()
        scene_rgb, wrist_rgb = _bgr_to_rgb(scene), _bgr_to_rgb(wrist)
        actions_model, dt_ms = self.client.act(
            scene_rgb, wrist_rgb, arm.prompt, state_model, cfg.num_steps
        )
        for _ in range(max(0, cfg.best_of - 1)):
            cand, dt_c = self.client.act(
                scene_rgb, wrist_rgb, arm.prompt, state_model, cfg.num_steps
            )
            dt_ms += dt_c
            if _grip_travel(cand) > _grip_travel(actions_model):
                actions_model = cand
        actions = np.clip(arm.to_arm_frame(actions_model), arm.joint_min, arm.joint_max)

        self._preds[arm.name] += 1
        n = self._preds[arm.name]
        if n <= cfg.warmup_predictions:
            arm.status = f"warm-up {n}/{cfg.warmup_predictions} ({dt_ms:.0f} ms)"
            print(f"[{arm.name}] warm-up prediction {n}/{cfg.warmup_predictions} ({dt_ms:.0f} ms)")
            return

        jump = np.abs(actions[0] - state_arm)
        if float(jump.max()) > arm.max_chunk_jump_deg:
            arm.status = f"DROPPED chunk: {jump.max():.0f} deg jump on joint {int(jump.argmax())}"
            print(
                f"[{arm.name}] DROPPED chunk: first action is {jump.max():.0f} deg away from the "
                f"current pose (joint {int(jump.argmax())}, limit {arm.max_chunk_jump_deg:.0f}).\n"
                f"    arm state   = {_fmt(state_arm)}\n"
                f"    model state = {_fmt(state_model)}\n"
                f"    action[0]   = {_fmt(actions[0])}\n"
                "    If this repeats, the joint offsets/signs are probably wrong for your "
                "calibration: run `so101_client.py check`."
            )
            return

        # Chunk step k is meant for t_obs + k/30 s. When inference takes longer
        # than the chunk (30 steps = 1 s) aligning strictly to t_obs would
        # expire it on arrival, so skip at most `max_latency_skip_s` of it.
        t_start = max(t_obs, time.monotonic() - cfg.max_latency_skip_s)
        arm.last_plan_peak = float(np.abs(actions - state_arm).max())
        arm.last_plan_t = time.monotonic()
        arm.recent_peaks.append(arm.last_plan_peak)
        chunk_id = arm.ring.add(actions, t_start)
        arm.status = (f"{'DRY RUN  ' if cfg.dry_run else ''}chunk {chunk_id}  {dt_ms:.0f} ms  "
                      f"plan end-state {_fmt(actions[-1] - state_arm)}  |  {arm.prompt}")
        print(f"[{arm.name}] {dt_ms:4.0f} ms  chunk {chunk_id}  "
              f"a0-state={_fmt(actions[0] - state_arm)}  end-state={_fmt(actions[-1] - state_arm)}")

    def run(self) -> None:
        while not self._stop_event.is_set():
            for arm in self.arms:
                if self._stop_event.is_set():
                    return
                try:
                    self._step(arm)
                except Exception as e:  # noqa: BLE001
                    print(f"[{arm.name}] inference error: {type(e).__name__}: {e}")
                    time.sleep(0.5)


class _ExecutionConsumer(threading.Thread):
    def __init__(self, arm: ArmRuntime, cfg: RuntimeConfig):
        super().__init__(daemon=True, name=f"ExecutionConsumer-{arm.name}")
        self.arm = arm
        self.cfg = cfg
        self.smooth_alpha = float(np.clip(cfg.smooth_alpha, 0.05, 1.0))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        arm, cfg = self.arm, self.cfg
        interval = 1.0 / cfg.exec_hz
        last_sent: np.ndarray | None = None
        holding = False
        next_tick = time.monotonic()
        current = None  # (chunk, t_start, chunk_id) followed in "commit" mode
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < arm.hold_until:
                next_tick = now + interval
                time.sleep(interval)
                continue
            entries = arm.ring.snapshot()
            active, ages = [], []
            if cfg.execution_mode == "commit":
                # Follow one chunk for `commit_steps` before switching to the
                # newest, so a single reaching plan isn't averaged away.
                if entries and (current is None
                                or int((now - current[1]) * ACTION_FPS) >= cfg.commit_steps):
                    current = entries[-1]
                if current is not None:
                    step = int((now - current[1]) * ACTION_FPS)
                    if 0 <= step < current[0].shape[0]:
                        active, ages = [current[0][step]], [0.0]
            else:
                for chunk, t_obs, _ in entries:
                    n = chunk.shape[0]
                    if cfg.actions_per_chunk is not None:
                        n = min(n, cfg.actions_per_chunk)
                    step = int((now - t_obs) * ACTION_FPS)
                    if 0 <= step < n:
                        active.append(chunk[step])
                        ages.append(now - t_obs)

            if active:
                holding = False
                w = np.exp(-cfg.ensemble_m * np.asarray(ages, dtype=np.float32))
                w /= w.sum()
                target = (w[:, None] * np.stack(active)).sum(axis=0).astype(np.float32)
                target = clip_step(target, arm.follower.get_state(), arm.max_step_deg)
                if last_sent is not None and self.smooth_alpha < 1.0:
                    target = self.smooth_alpha * target + (1.0 - self.smooth_alpha) * last_sent
                last_sent = target
                arm.follower.set_target(target)
            elif not holding:
                print(f"[{arm.name}] no active chunk, holding position")
                holding = True

            next_tick += interval
            time.sleep(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() - next_tick > 1.0:  # fell far behind; resync
                next_tick = time.monotonic()


class AsyncPolicyRunner:
    def __init__(self, client: PolicyClient, arms: list[ArmRuntime], cfg: RuntimeConfig):
        self._producer = _InferenceProducer(client, arms, cfg)
        self._consumers = [] if cfg.dry_run else [_ExecutionConsumer(a, cfg) for a in arms]

    def __enter__(self) -> "AsyncPolicyRunner":
        self._producer.start()
        for c in self._consumers:
            c.start()
        return self

    def __exit__(self, *exc) -> bool:
        for c in self._consumers:
            c.stop()
        self._producer.stop()
        for c in self._consumers:
            c.join(timeout=2.0)
        self._producer.join(timeout=5.0)
        return False

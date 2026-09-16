"""Robot-side client: drive one or two SO-101 arms with MolmoAct2-SO100_101.

Talks to `host_server_so101.py` over HTTP (same machine or LAN). All hardware
wiring — ports, calibration ids, cameras, joint conventions, safety limits —
lives in a YAML config (see `configs/`).

Subcommands (run from examples/so101/):

    uv run so101_client.py cameras                       # list webcams + RealSense serials
    uv run so101_client.py cameras --config configs/single_arm.yaml --snapshot snaps/
    uv run so101_client.py check --config configs/single_arm.yaml [--limp]
    uv run so101_client.py run   --config configs/single_arm.yaml --prompt "pick up the lemon" --dry-run
    uv run so101_client.py run   --config configs/two_arms.yaml \
        --arm-prompt left="pick up the red block" --arm-prompt right="pick up the blue block"

MolmoAct2-SO100_101 is a *single-arm* policy. With two arms each arm gets its
own independent query (own wrist camera, own state, own prompt) — there is no
coordination between them, so keep their workspaces apart.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import yaml

from so101_hardware import (
    JOINT_COUNT, MOTOR_NAMES, FollowerArm, make_camera, wait_for_frames,
)
from so101_preview import PreviewServer
from so101_runtime import ArmRuntime, AsyncPolicyRunner, PolicyClient, RuntimeConfig

HERE = os.path.dirname(os.path.abspath(__file__))

# LeRobot v3.0 (current calibration) -> v2.1 (MolmoAct2 training data):
#   shoulder_lift: old = 90 - new ; elbow_flex: old = new + 90
# https://huggingface.co/docs/lerobot/backwardcomp
DEFAULT_OFFSETS = [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]
DEFAULT_SIGNS = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]

# Fallback copy of the checkpoint's state quantiles (norm_stats.json,
# tag so100_so101_molmoact2) for `check` when the server isn't up.
TRAIN_STATE_Q01 = [-41.9, 43.7, 38.4, 5.7, -63.4, 0.9]
TRAIN_STATE_Q99 = [48.3, 185.3, 173.1, 91.8, 42.9, 44.1]
# Model-card sample: an SO-100 in its folded rest pose, in model frame.
REST_POSE_MODEL = [-0.5, 189.1, 181.4, 60.6, -3.6, 1.1]


# ─── config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get("arms"):
        raise SystemExit(f"{path}: needs at least one entry under `arms:`")
    cams = cfg.get("cameras") or {}
    for arm in cfg["arms"]:
        for key in ("name", "port", "calibration_id", "scene_camera"):
            if not arm.get(key):
                raise SystemExit(f"{path}: arm {arm.get('name', '?')!r} is missing `{key}`")
        for key in ("scene_camera", "wrist_camera"):
            if arm.get(key) and arm[key] not in cams:
                raise SystemExit(f"{path}: arm {arm['name']!r}: {key} {arm[key]!r} "
                                 f"is not defined under `cameras:`")
        if not arm.get("wrist_camera") and not arm.get("scene_only"):
            raise SystemExit(f"{path}: arm {arm['name']!r} has no wrist_camera; "
                             "set one or set `scene_only: true`")
    return cfg


def _vec(arm: dict[str, Any], key: str, default: list[float] | float) -> np.ndarray:
    val = arm.get(key, default)
    if val is None:
        val = default
    if np.isscalar(val):
        val = [val] * JOINT_COUNT
    val = [np.nan if v is None else float(v) for v in val]
    if len(val) != JOINT_COUNT:
        raise SystemExit(f"arm {arm['name']!r}: `{key}` needs {JOINT_COUNT} values")
    return np.asarray(val, dtype=np.float32)


def _joint_frame(arm: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    signs = _vec(arm, "joint_signs", DEFAULT_SIGNS)
    if not np.all(np.isin(signs, [-1.0, 1.0])):
        raise SystemExit(f"arm {arm['name']!r}: joint_signs must be +1/-1")
    return signs, _vec(arm, "joint_offsets", DEFAULT_OFFSETS)


def _open_cameras(cfg: dict[str, Any], names: set[str]) -> dict[str, Any]:
    cams: dict[str, Any] = {}
    try:
        for name in sorted(names):
            cams[name] = make_camera(name, cfg["cameras"][name])
    except Exception:
        for c in cams.values():
            c.close()
        raise
    return cams


def _used_cameras(cfg: dict[str, Any]) -> set[str]:
    names = set()
    for arm in cfg["arms"]:
        names.add(arm["scene_camera"])
        if arm.get("wrist_camera") and not arm.get("scene_only"):
            names.add(arm["wrist_camera"])
    return names


def _open_arms(cfg: dict[str, Any], simulate: bool) -> list[FollowerArm]:
    arms: list[FollowerArm] = []
    try:
        for a in cfg["arms"]:
            arms.append(FollowerArm(
                name=a["name"], port=a["port"], calibration_id=a["calibration_id"],
                calibration_dir=a.get("calibration_dir"), simulate=simulate,
            ))
    except Exception:
        for arm in arms:
            arm.disconnect()
        raise
    return arms


# ─── cameras ─────────────────────────────────────────────────────────────────

def cmd_cameras(args: argparse.Namespace) -> None:
    print("V4L2 cameras (use the by-id path in your config; /dev/videoN can change between boots):")
    by_id = sorted(glob.glob("/dev/v4l/by-id/*"))
    if not by_id:
        print("  (none)")
    for p in by_id:
        print(f"  {p}\n      -> {os.path.realpath(p)}")
    print("  For each physical camera, the `...-video-index0` node is the image stream.")
    try:
        import pyrealsense2 as rs

        devs = list(rs.context().query_devices())
        print("RealSense devices:" if devs else "RealSense devices: (none)")
        for d in devs:
            print(f"  {d.get_info(rs.camera_info.name)}  serial={d.get_info(rs.camera_info.serial_number)}"
                  f"  usb={d.get_info(rs.camera_info.usb_type_descriptor)}")
    except ImportError:
        print("RealSense devices: pyrealsense2 not installed")

    if not args.config:
        return
    cfg = load_config(args.config)
    cams = _open_cameras(cfg, set(cfg.get("cameras") or {}))
    try:
        wait_for_frames(cams)
        if args.snapshot:
            os.makedirs(args.snapshot, exist_ok=True)
            for name, cam in cams.items():
                path = os.path.join(args.snapshot, f"{name}.jpg")
                cv2.imwrite(path, cam.read())
                print(f"saved {path}")
        if args.show:
            preview = PreviewServer(cams, port=args.preview_port)
            print("Ctrl+C to stop")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            preview.close()
    finally:
        for c in cams.values():
            c.close()


# ─── check ───────────────────────────────────────────────────────────────────

def _train_ranges(server: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    if server:
        try:
            h = PolicyClient(server).health()
            if h.get("state_q01") and h.get("state_q99"):
                return (np.asarray(h["state_q01"], np.float32), np.asarray(h["state_q99"], np.float32),
                        f"{h.get('repo_id')} via {server}")
        except Exception as e:  # noqa: BLE001
            print(f"(server {server} not reachable: {e}; using built-in training ranges)")
    return (np.asarray(TRAIN_STATE_Q01, np.float32), np.asarray(TRAIN_STATE_Q99, np.float32),
            "built-in copy of norm_stats.json")


def cmd_check(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    q01, q99, src = _train_ranges(args.server or cfg.get("server"))
    margin = 20.0
    print(f"Training state range (q01..q99) from {src}")
    followers = _open_arms(cfg, simulate=False)
    frames = {a["name"]: _joint_frame(a) for a in cfg["arms"]}
    if args.limp:
        print("\n!!! --limp: disabling torque in 3 s. Support the arm(s) or they will drop. !!!")
        time.sleep(3.0)
        for f in followers:
            f.request_torque(False)
    print("\nPut each arm in its folded REST pose (the pose you calibrated from / power off in).")
    print(f"Expected model-frame rest pose is roughly {REST_POSE_MODEL}.")
    print("Ctrl+C to stop.\n")
    try:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            for f in followers:
                signs, offsets = frames[f.name]
                arm_state = f.get_state()
                model = signs * arm_state + offsets
                lines = [f"── {f.name} " + "─" * 60,
                         f"{'joint':<14}{'arm (lerobot)':>14}{'model frame':>13}{'train q01..q99':>20}"]
                bad = []
                for i, n in enumerate(MOTOR_NAMES):
                    out = model[i] < q01[i] - margin or model[i] > q99[i] + margin
                    if out:
                        bad.append(n)
                    lines.append(f"{n:<14}{arm_state[i]:>14.1f}{model[i]:>13.1f}"
                                 f"{f'{q01[i]:.0f}..{q99[i]:.0f}':>20}{'  <-- OUT OF RANGE' if out else ''}")
                print("\n".join(lines))
                if bad:
                    print(f"   {len(bad)} joint(s) far outside the training distribution: {bad}. "
                          "Move the arm into the workspace; if a joint stays out of range, "
                          "its joint_offsets/joint_signs are wrong.")
            print()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for f in followers:
            f.disconnect()


# ─── selftest ────────────────────────────────────────────────────────────────

def cmd_selftest(args: argparse.Namespace) -> None:
    """Move each joint +amp, -amp and back, slowly, one at a time — no model.

    Confirms the command path (bus, torque, calibration, rate limiting) and
    reports how far each joint actually travelled.
    """
    cfg = load_config(args.config)
    a = next((x for x in cfg["arms"] if x["name"] == args.arm), cfg["arms"][0])
    joints = [j.strip() for j in args.joints.split(",")]
    unknown = [j for j in joints if j not in MOTOR_NAMES]
    if unknown:
        raise SystemExit(f"unknown joints {unknown}; choose from {MOTOR_NAMES}")
    follower = FollowerArm(name=a["name"], port=a["port"], calibration_id=a["calibration_id"],
                           calibration_dir=a.get("calibration_dir"))
    hz = 30.0

    def ramp(start: np.ndarray, goal: np.ndarray) -> None:
        steps = max(1, int(np.abs(goal - start).max() / args.speed * hz))
        for k in range(1, steps + 1):
            follower.set_target(start + (goal - start) * k / steps)
            time.sleep(1.0 / hz)
        time.sleep(0.4)  # let the servo settle

    try:
        home = follower.get_state()
        print(f"[selftest] start pose {np.round(home, 1).tolist()}; "
              f"±{args.amp:.0f} deg at {args.speed:.0f} deg/s. Ctrl+C to abort.")
        for j in joints:
            i = MOTOR_NAMES.index(j)
            travel = []
            for sign in (+1, -1):
                goal = home.copy()
                goal[i] += sign * args.amp
                ramp(follower.get_state(), goal)
                travel.append(float(follower.get_state()[i] - home[i]))
            ramp(follower.get_state(), home)
            ok = travel[0] > args.amp * 0.5 and travel[1] < -args.amp * 0.5
            print(f"[selftest] {j:<14} moved {travel[0]:+5.1f} / {travel[1]:+5.1f} deg  "
                  f"{'OK' if ok else 'DID NOT MOVE AS COMMANDED'}")
    except KeyboardInterrupt:
        print("[selftest] aborted")
    finally:
        follower.disconnect(disable_torque=not cfg.get("hold_torque_on_exit", False))


# ─── demo ────────────────────────────────────────────────────────────────────

# Waypoints as offsets (deg, arm frame) from the starting folded pose:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
_DEMO_WAYPOINTS = [
    ("lift up",          [0,   35, -30, -30,   0,  0]),
    ("open gripper",     [0,   35, -30, -30,   0, 35]),
    ("sweep one way",    [25,  35, -30, -30,   0, 35]),
    ("sweep other way",  [-25, 35, -30, -30,   0, 35]),
    ("center",           [0,   35, -30, -30,   0, 35]),
    ("twist wrist",      [0,   35, -30, -30,  60, 35]),
    ("twist back",       [0,   35, -30, -30, -60, 35]),
    ("wrist straight",   [0,   35, -30, -30,   0, 35]),
    ("close gripper",    [0,   35, -30, -30,   0,  0]),
    ("open gripper",     [0,   35, -30, -30,   0, 35]),
    ("close gripper",    [0,   35, -30, -30,   0,  0]),
    ("fold back down",   [0,    0,   0,   0,   0,  0]),
]


def cmd_demo(args: argparse.Namespace) -> None:
    """Scripted, model-free motion demo from the current (folded) pose."""
    cfg = load_config(args.config)
    a = next((x for x in cfg["arms"] if x["name"] == args.arm), cfg["arms"][0])
    follower = FollowerArm(name=a["name"], port=a["port"], calibration_id=a["calibration_id"],
                           calibration_dir=a.get("calibration_dir"))
    hz = 30.0
    try:
        home = follower.get_state()
        print(f"[demo] start pose {np.round(home, 1).tolist()}  scale={args.scale}  speed={args.speed} deg/s")
        for _ in range(args.repeat):
            for label, off in _DEMO_WAYPOINTS:
                start = follower.get_state()
                goal = home + np.asarray(off, np.float32) * args.scale
                steps = max(1, int(np.abs(goal - start).max() / args.speed * hz))
                print(f"[demo] {label}")
                for k in range(1, steps + 1):
                    follower.set_target(start + (goal - start) * k / steps)
                    time.sleep(1.0 / hz)
                time.sleep(0.3)
        print(f"[demo] done; final pose {np.round(follower.get_state(), 1).tolist()}")
    except KeyboardInterrupt:
        print("[demo] aborted")
    finally:
        follower.disconnect(disable_torque=not cfg.get("hold_torque_on_exit", False))


# ─── probe ───────────────────────────────────────────────────────────────────

def cmd_probe(args: argparse.Namespace) -> None:
    """Query the model on live cameras + live joint state without moving the arm.

    Reads positions straight off the bus (no configure(), so torque is left as
    it is) and overlays the planned reach on the browser preview. Useful for
    placing the scene camera: move it until the planned motion jumps.
    """
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = load_config(args.config)
    a = next((x for x in cfg["arms"] if x["name"] == args.arm), cfg["arms"][0])
    client = PolicyClient(args.server or cfg.get("server", "http://127.0.0.1:8101"))
    client.health()
    prompt = args.prompt or a.get("prompt") or cfg.get("prompt")
    signs, offsets = _joint_frame(a)
    use_scene_only = bool(a.get("scene_only", False))
    names = {a["scene_camera"]} | (set() if use_scene_only else {a["wrist_camera"]})
    robot = None
    if args.state is None:
        robot = SOFollower(SOFollowerRobotConfig(port=a["port"], id=a["calibration_id"]))
        robot.bus.connect()
    else:
        fixed_state = np.array([float(x) for x in args.state.split(",")], np.float32)
        if fixed_state.shape != (JOINT_COUNT,):
            raise SystemExit(f"--state needs {JOINT_COUNT} comma-separated model-frame values")
        print(f"[probe] using fixed model-frame state {fixed_state.tolist()} (arm not read)")
    cams = _open_cameras(cfg, names)
    status = ["waiting for first prediction"]
    preview = PreviewServer(cams, port=args.preview_port, status_fn=lambda: status)
    print(f"[probe] arm {a['name']!r} never moves. prompt={prompt!r}. Ctrl+C to stop.")
    try:
        wait_for_frames(cams)
        while True:
            if robot is None:
                state = fixed_state
            else:
                pos = robot.bus.sync_read("Present_Position")
                state = signs * np.array([pos[n] for n in MOTOR_NAMES], np.float32) + offsets
            scene = cams[a["scene_camera"]].read()
            wrist = scene if use_scene_only else cams[a["wrist_camera"]].read()
            actions, dt_ms = client.act(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB),
                                        cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB),
                                        prompt, state, int(cfg.get("num_steps", 10)))
            delta = actions - state
            peak = float(np.abs(delta).max())
            line = (f"planned reach {peak:5.1f} deg  ({'MOVING' if peak > 15 else 'barely'})  "
                    f"end delta {np.round(delta[-1]).astype(int).tolist()}  {dt_ms:.0f} ms")
            status[:] = [line, prompt]
            print(f"[probe] {line}")
    except KeyboardInterrupt:
        pass
    finally:
        preview.close()
        if robot is not None:
            robot.bus.disconnect(False)
        for c in cams.values():
            c.close()


# ─── run ─────────────────────────────────────────────────────────────────────

def _rehome_supervisor(arms: list[ArmRuntime], arm_cfgs: list[dict[str, Any]],
                       pose: str, after_s: float, health: dict[str, Any],
                       stall_deg: float = 6.0, every_s: float = 0.0) -> None:
    """Re-home an arm the policy has parked outside its training state range.

    From the folded rest pose the checkpoint plans nothing but a small pull back
    toward the range, so the run would stall there forever.
    """
    q01 = np.asarray(health.get("state_q01") or TRAIN_STATE_Q01, np.float32)
    q99 = np.asarray(health.get("state_q99") or TRAIN_STATE_Q99, np.float32)
    # No slack: the checkpoint only knows states inside its training range, and
    # at the boundary it stops planning grasps (gripper deltas collapse to ~0).
    margin = 0.0
    out_since: dict[str, float] = {}
    last_home = {a.name: time.monotonic() for a in arms}
    while True:
        time.sleep(1.0)
        for arm, a in zip(arms, arm_cfgs):
            if time.monotonic() < arm.hold_until:
                continue
            # Periodic reset: an attempt takes ~20-40 s, and the policy is most
            # capable right after landing in an in-distribution pose.
            due = every_s > 0 and time.monotonic() - last_home[arm.name] >= every_s
            # Treat "within 2 deg of a limit" as out too; the arm creeps there.
            edge = 2.0
            state = arm.to_model_frame(arm.follower.get_state())
            out = bool(np.any(state < q01 + edge) or np.any(state > q99 - edge))
            # A policy that has stalled plans almost nothing for many cycles in
            # a row; re-homing gives it a fresh pose it knows what to do from.
            # Median over the recent plans: one stray big plan should not reset
            # the stall timer, which made this almost never fire.
            peaks = list(arm.recent_peaks)
            stalled = (arm.last_plan_t > 0 and time.monotonic() - arm.last_plan_t < 5.0
                       and len(peaks) >= 10 and float(np.median(peaks[-10:])) < stall_deg)
            out = out or stalled or due
            if not out:
                out_since.pop(arm.name, None)
                continue
            t0 = out_since.setdefault(arm.name, time.monotonic())
            if time.monotonic() - t0 < after_s:
                continue
            bad = [i for i, v in enumerate(state) if v < q01[i] + edge or v > q99[i] - edge]
            why = ("periodic reset" if due and not bad and not stalled
                   else f"plans stalled under {stall_deg:.0f} deg" if not bad
                   else f"outside training range on {[MOTOR_NAMES[i] for i in bad]}")
            print(f"[{arm.name}] {why} for {after_s:.0f}s; re-homing to the start pose")
            arm.hold_until = time.monotonic() + 60.0
            arm.status = "re-homing to start pose"
            arm.ring.clear()          # stale plans belong to the pose we are leaving
            try:
                _ramp_to_model_pose(arm.follower, a, pose)
            except Exception as e:  # noqa: BLE001
                print(f"[{arm.name}] re-home failed: {e}")
            out_since.pop(arm.name, None)
            last_home[arm.name] = time.monotonic()
            arm.ring.clear()
            arm.recent_peaks.clear()
            arm.hold_until = time.monotonic() + 0.5


def _ramp_to_model_pose(follower: FollowerArm, arm: dict[str, Any], pose: str,
                        speed_deg_s: float = 30.0) -> None:
    target = np.array([float(x) for x in pose.split(",")], np.float32)
    if target.shape != (JOINT_COUNT,):
        raise SystemExit(f"--start-pose needs {JOINT_COUNT} comma-separated values")
    signs, offsets = _joint_frame(arm)
    goal = (target - offsets) * signs
    start = follower.get_state()
    steps = max(1, int(np.abs(goal - start).max() / speed_deg_s * 30))
    print(f"[{arm['name']}] ramping to start pose {np.round(goal, 1).tolist()} (arm frame)")
    for k in range(1, steps + 1):
        follower.set_target(start + (goal - start) * k / steps)
        time.sleep(1 / 30)
    time.sleep(0.8)

def _parse_arm_prompts(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--arm-prompt expects NAME=TEXT, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    server = args.server or cfg.get("server", "http://127.0.0.1:8101")
    arm_prompts = _parse_arm_prompts(args.arm_prompt)
    unknown = set(arm_prompts) - {a["name"] for a in cfg["arms"]}
    if unknown:
        raise SystemExit(f"--arm-prompt for unknown arm(s): {sorted(unknown)}")

    prompts = {}
    for a in cfg["arms"]:
        p = arm_prompts.get(a["name"]) or args.prompt or a.get("prompt") or cfg.get("prompt")
        if not p:
            raise SystemExit(f"no prompt for arm {a['name']!r}: pass --prompt or --arm-prompt")
        prompts[a["name"]] = p

    client = PolicyClient(server)
    try:
        h = client.health()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot reach policy server at {server}: {e}\n"
                         "Start it with ./run_so101.sh from the repo root.") from e
    if h.get("state_dim") != JOINT_COUNT:
        raise SystemExit(f"server at {server} serves {h.get('repo_id')} "
                         f"(state_dim={h.get('state_dim')}), not the SO-100/101 checkpoint")
    print(f"[client] server {server}: {h.get('repo_id')} ({h.get('dtype')})")

    rt_cfg = RuntimeConfig(
        exec_hz=float(cfg.get("exec_hz", 30.0)),
        num_steps=int(args.num_steps or cfg.get("num_steps", 10)),
        ensemble_m=float(cfg.get("ensemble_m", 0.5)),
        smooth_alpha=float(cfg.get("smooth_alpha", 1.0)),
        actions_per_chunk=cfg.get("actions_per_chunk"),
        warmup_predictions=int(cfg.get("warmup_predictions", 2)),
        max_latency_skip_s=float(cfg.get("max_latency_skip_s", 0.3)),
        execution_mode=str(cfg.get("execution_mode", "ensemble")),
        commit_steps=int(cfg.get("commit_steps", 15)),
        best_of=int(getattr(args, "best_of", None) or cfg.get("best_of", 1)),
        save_frames_dir=args.save_frames_dir,
        dry_run=args.dry_run,
    )
    if rt_cfg.execution_mode not in ("ensemble", "commit"):
        raise SystemExit("execution_mode must be 'ensemble' or 'commit'")
    if rt_cfg.save_frames_dir:
        os.makedirs(rt_cfg.save_frames_dir, exist_ok=True)

    cams = _open_cameras(cfg, _used_cameras(cfg))
    followers: list[FollowerArm] = []
    stop_once = threading.Lock()

    def cleanup() -> None:
        if not stop_once.acquire(blocking=False):
            return
        release = not cfg.get("hold_torque_on_exit", False)
        print(f"\n[client] shutting down ({'releasing torque' if release else 'holding torque'})")
        for f in followers:
            try:
                f.disconnect(disable_torque=release)
            except Exception as e:  # noqa: BLE001
                print(f"[client] {f.name} disconnect error: {e}")
        for c in cams.values():
            c.close()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        wait_for_frames(cams)
        followers.extend(_open_arms(cfg, simulate=args.simulate))
        if args.start_pose:
            for a, f in zip(cfg["arms"], followers):
                _ramp_to_model_pose(f, a, args.start_pose)
        arms = []
        for a, f in zip(cfg["arms"], followers):
            signs, offsets = _joint_frame(a)
            arms.append(ArmRuntime(
                name=a["name"], follower=f,
                scene_cam=cams[a["scene_camera"]],
                wrist_cam=None if a.get("scene_only") else cams[a["wrist_camera"]],
                prompt=prompts[a["name"]],
                signs=signs, offsets=offsets,
                joint_min=np.nan_to_num(_vec(a, "joint_min", -np.inf), nan=-np.inf),
                joint_max=np.nan_to_num(_vec(a, "joint_max", np.inf), nan=np.inf),
                max_step_deg=float(a.get("max_step_deg", 5.0)),
                max_chunk_jump_deg=float(a.get("max_chunk_jump_deg", 30.0)),
                scene_only=bool(a.get("scene_only", False)),
                prompt_file=args.prompt_file,
            ))
            print(f"[client] arm {a['name']!r}: prompt={prompts[a['name']]!r}")

        if args.dry_run:
            print("[client] --dry-run: predictions only, arms will NOT move")
        if args.start_pose and args.rehome_after > 0:
            threading.Thread(target=_rehome_supervisor, daemon=True, name="rehome",
                             args=(arms, cfg["arms"], args.start_pose, args.rehome_after,
                                   client.health(), args.stall_deg, args.rehome_every)).start()
        if args.show:
            PreviewServer(cams, port=args.preview_port,
                          status_fn=lambda: [f"{a.name}: {a.status}" for a in arms])
        print("[client] Ctrl+C to stop")

        with AsyncPolicyRunner(client, arms, rt_cfg):
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cameras", help="list cameras; with --config, open them and optionally snapshot/show")
    c.add_argument("--config")
    c.add_argument("--snapshot", metavar="DIR", help="save one frame per configured camera")
    c.add_argument("--show", action="store_true", help="live browser preview (see --preview-port)")
    c.add_argument("--preview-port", type=int, default=8102)
    c.set_defaults(func=cmd_cameras)

    k = sub.add_parser("check", help="print joint states in arm + model frame vs. training ranges")
    k.add_argument("--config", required=True)
    k.add_argument("--server", help="override `server:` from the config")
    k.add_argument("--seconds", type=float, default=600.0)
    k.add_argument("--limp", action="store_true", help="disable torque so the arm can be moved by hand")
    k.set_defaults(func=cmd_check)

    st = sub.add_parser("selftest", help="move each joint a little and back (no model)")
    st.add_argument("--config", required=True)
    st.add_argument("--arm", help="arm name (default: first arm)")
    st.add_argument("--joints", default="gripper,wrist_roll,wrist_flex,shoulder_pan",
                    help=f"comma-separated subset of {MOTOR_NAMES}")
    st.add_argument("--amp", type=float, default=10.0, help="degrees each way (default 10)")
    st.add_argument("--speed", type=float, default=20.0, help="deg/s (default 20)")
    st.set_defaults(func=cmd_selftest)

    dm = sub.add_parser("demo", help="scripted motion demo: lift, sweep, twist, grip, fold (no model)")
    dm.add_argument("--config", required=True)
    dm.add_argument("--arm", help="arm name (default: first arm)")
    dm.add_argument("--scale", type=float, default=1.0, help="multiply all demo motions (default 1.0)")
    dm.add_argument("--speed", type=float, default=45.0, help="deg/s (default 45)")
    dm.add_argument("--repeat", type=int, default=1)
    dm.set_defaults(func=cmd_demo)

    pr = sub.add_parser("probe", help="live model predictions on the cameras without moving the arm")
    pr.add_argument("--config", required=True)
    pr.add_argument("--server", help="override `server:` from the config")
    pr.add_argument("--arm", help="arm name (default: first arm)")
    pr.add_argument("--prompt")
    pr.add_argument("--state", help="fixed model-frame state 'a,b,c,d,e,f' instead of reading the arm")
    pr.add_argument("--preview-port", type=int, default=8102)
    pr.set_defaults(func=cmd_probe)

    r = sub.add_parser("run", help="run the policy")
    r.add_argument("--config", required=True)
    r.add_argument("--server", help="override `server:` from the config")
    r.add_argument("--prompt", help="instruction for every arm")
    r.add_argument("--arm-prompt", action="append", default=[], metavar="NAME=TEXT",
                   help="per-arm instruction (repeatable); wins over --prompt")
    r.add_argument("--num-steps", type=int, help="flow-matching solver steps (default 10)")
    r.add_argument("--best-of", type=int,
                   help="draw N plans per cycle, keep the one that commits most to a grasp "
                        "(default 1; costs N x inference latency)")
    r.add_argument("--dry-run", action="store_true", help="query the model but never move the arms")
    r.add_argument("--simulate", action="store_true", help="no serial I/O; fake arms that track targets")
    r.add_argument("--show", action="store_true",
                   help="serve a live camera + status view at http://127.0.0.1:PREVIEW_PORT/")
    r.add_argument("--preview-port", type=int, default=8102)
    r.add_argument("--save-frames-dir", help="save every image sent to the model")
    r.add_argument("--start-pose", metavar="MODEL_FRAME_POSE",
                   help="ramp to this model-frame pose 'a,b,c,d,e,f' before the policy starts "
                        "(the folded rest pose is outside the training range)")
    r.add_argument("--prompt-file", help="re-read the prompt from this file whenever it changes")
    r.add_argument("--rehome-every", type=float, default=0.0, metavar="SECONDS",
                   help="with --start-pose: also reset on this fixed cycle, so every attempt "
                        "starts from a pose the policy handles well (0 = only on stall)")
    r.add_argument("--stall-deg", type=float, default=6.0, metavar="DEG",
                   help="plans smaller than this count as stalled (see --rehome-after)")
    r.add_argument("--rehome-after", type=float, default=15.0, metavar="SECONDS",
                   help="with --start-pose: if the policy parks the arm outside the training "
                        "state range for this long, ramp back to the start pose (0 disables)")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    if getattr(args, "config", None) and not os.path.exists(args.config):
        alt = os.path.join(HERE, args.config)
        if os.path.exists(alt):
            args.config = alt
    args.func(args)


if __name__ == "__main__":
    main()

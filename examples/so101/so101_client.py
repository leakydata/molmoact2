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
            print("Showing cameras, press q to quit")
            while True:
                for name, cam in cams.items():
                    img = cam.read()
                    if img is not None:
                        cv2.imshow(name, img)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
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


# ─── run ─────────────────────────────────────────────────────────────────────

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
        save_frames_dir=args.save_frames_dir,
        dry_run=args.dry_run,
    )
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
            ))
            print(f"[client] arm {a['name']!r}: prompt={prompts[a['name']]!r}")

        if args.dry_run:
            print("[client] --dry-run: predictions only, arms will NOT move")
        print("[client] Ctrl+C to stop" + (" (or q in a preview window)" if args.show else ""))

        with AsyncPolicyRunner(client, arms, rt_cfg):
            while True:
                if args.show:
                    for name, cam in cams.items():
                        img = cam.read()
                        if img is not None:
                            cv2.imshow(name, img)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        if args.show:
            cv2.destroyAllWindows()


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cameras", help="list cameras; with --config, open them and optionally snapshot/show")
    c.add_argument("--config")
    c.add_argument("--snapshot", metavar="DIR", help="save one frame per configured camera")
    c.add_argument("--show", action="store_true", help="live preview windows")
    c.set_defaults(func=cmd_cameras)

    k = sub.add_parser("check", help="print joint states in arm + model frame vs. training ranges")
    k.add_argument("--config", required=True)
    k.add_argument("--server", help="override `server:` from the config")
    k.add_argument("--seconds", type=float, default=600.0)
    k.add_argument("--limp", action="store_true", help="disable torque so the arm can be moved by hand")
    k.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="run the policy")
    r.add_argument("--config", required=True)
    r.add_argument("--server", help="override `server:` from the config")
    r.add_argument("--prompt", help="instruction for every arm")
    r.add_argument("--arm-prompt", action="append", default=[], metavar="NAME=TEXT",
                   help="per-arm instruction (repeatable); wins over --prompt")
    r.add_argument("--num-steps", type=int, help="flow-matching solver steps (default 10)")
    r.add_argument("--dry-run", action="store_true", help="query the model but never move the arms")
    r.add_argument("--simulate", action="store_true", help="no serial I/O; fake arms that track targets")
    r.add_argument("--show", action="store_true", help="camera preview windows")
    r.add_argument("--save-frames-dir", help="save every image sent to the model")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    if getattr(args, "config", None) and not os.path.exists(args.config):
        alt = os.path.join(HERE, args.config)
        if os.path.exists(alt):
            args.config = alt
    args.func(args)


if __name__ == "__main__":
    main()

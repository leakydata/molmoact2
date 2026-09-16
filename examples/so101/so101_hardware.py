"""Robot-side hardware for the SO-101 MolmoAct2 client.

* `FollowerArm` wraps LeRobot's `SOFollower` driver. Serial I/O runs on one
  background thread per arm (the Feetech bus is half-duplex), so callers only
  touch lock-protected target/state slots.
* `OpenCVCamera` / `RealSenseCamera` each keep the latest frame from a
  background grab thread, so a scene camera can be shared by two arms and the
  inference loop never blocks on a USB read.

Frames are stored as BGR uint8 (OpenCV convention) and converted to RGB right
before they are sent to the server.

Adapted from https://github.com/irenegracekp/molmoact2-so101.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_COUNT = len(MOTOR_NAMES)
# Inner rate limiter on what is actually written to the servos, per worker
# iteration (~50-100 Hz on a healthy bus). A second, coarser limit
# (`max_step_deg`) is applied per control tick in the runtime.
_MAX_WRITE_DELTA_DEG = 4.0

_FLIP_CODES = {"v": 0, "h": 1, "180": -1, "none": None, None: None}


class FollowerArm:
    """Drives one SO-100/101 follower arm via LeRobot's `SOFollower`.

    `calibration_id` is the LeRobot robot id used at calibration time, i.e.
    the file `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`.
    """

    def __init__(self, name: str, port: str, calibration_id: str,
                 calibration_dir: str | None = None, simulate: bool = False):
        self.name = name
        self.simulate = simulate
        self._target = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._state = np.zeros(JOINT_COUNT, dtype=np.float32)
        self._target_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._torque_lock = threading.Lock()
        self._torque_desired = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.robot = None

        if simulate:
            print(f"[{name}] simulation mode (no serial I/O)")
            return

        # Import lazily so `--simulate` works without LeRobot installed.
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg_kwargs: dict[str, Any] = dict(port=port, id=calibration_id, use_degrees=True)
        if calibration_dir:
            cfg_kwargs["calibration_dir"] = Path(calibration_dir).expanduser()
        self.robot = SOFollower(SOFollowerRobotConfig(**cfg_kwargs))
        if not self.robot.calibration:
            raise RuntimeError(
                f"[{name}] no LeRobot calibration found for id {calibration_id!r} "
                f"(looked in {self.robot.calibration_fpath}). Run "
                f"`lerobot-calibrate --robot.type=so101_follower --robot.port={port} "
                f"--robot.id={calibration_id}` first."
            )
        # Servos on a sagging supply occasionally miss the connect-time motor
        # check or a config write; a short retry rides through that.
        for attempt in range(6):
            try:
                self.robot.connect(calibrate=False)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"[{name}] connect attempt {attempt + 1} failed ({e.__class__.__name__}); retrying")
                try:
                    self.robot.bus.disconnect(False)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(1.0)
        if not self.robot.is_calibrated:
            print(f"[{name}] motor calibration differs from {self.robot.calibration_fpath}; "
                  "writing file calibration to the motors")
            self.robot.bus.write_calibration(self.robot.calibration)
        init = self._read_state()
        # connect() re-enables torque; pin the goal to where the arm already is
        # so a stale Goal_Position register can't make it jump.
        self.robot.bus.sync_write(
            "Goal_Position", {n: float(init[i]) for i, n in enumerate(MOTOR_NAMES)}
        )
        self._target = init.copy()
        self._state = init.copy()
        print(f"[{name}] connected on {port} (calibration id {calibration_id!r})")
        self._thread = threading.Thread(target=self._worker_loop, daemon=True,
                                        name=f"arm-{name}")
        self._thread.start()

    def _read_state(self) -> np.ndarray:
        obs = self.robot.bus.sync_read("Present_Position")
        return np.array([obs[n] for n in MOTOR_NAMES], dtype=np.float32)

    def _worker_loop(self) -> None:
        torque_actual = True
        last_written = self._target.copy()
        while not self._stop.is_set():
            with self._torque_lock:
                desired = self._torque_desired
            if desired != torque_actual:
                try:
                    if desired:
                        last_written = self._read_state()
                        with self._target_lock:
                            self._target = last_written.copy()
                        self.robot.bus.enable_torque()
                    else:
                        self.robot.bus.disable_torque()
                    torque_actual = desired
                    print(f"[{self.name}] torque {'enabled' if desired else 'disabled'}")
                except Exception as e:  # noqa: BLE001
                    print(f"[{self.name}] torque transition error: {e}")

            if torque_actual:
                with self._target_lock:
                    target = self._target.copy()
                delta = np.clip(target - last_written, -_MAX_WRITE_DELTA_DEG, _MAX_WRITE_DELTA_DEG)
                last_written = last_written + delta
                try:
                    self.robot.bus.sync_write(
                        "Goal_Position",
                        {n: float(last_written[i]) for i, n in enumerate(MOTOR_NAMES)},
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[{self.name}] write error: {e}")

            try:
                state = self._read_state()
                with self._state_lock:
                    self._state = state
            except Exception as e:  # noqa: BLE001
                print(f"[{self.name}] read error: {e}")
                time.sleep(0.01)

    def set_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32)
        if self.simulate:
            with self._target_lock:
                delta = np.clip(target - self._target, -_MAX_WRITE_DELTA_DEG, _MAX_WRITE_DELTA_DEG)
                self._target = self._target + delta
            with self._state_lock:
                self._state = self._target.copy()
            return
        with self._target_lock:
            self._target = target.copy()

    def get_state(self) -> np.ndarray:
        with self._state_lock:
            return self._state.copy()

    def request_torque(self, on: bool) -> None:
        if self.simulate:
            return
        with self._torque_lock:
            self._torque_desired = on

    def disconnect(self, disable_torque: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.robot is not None and self.robot.bus.is_connected:
            # With torque off the arm goes limp and drops onto whatever is below it.
            self.robot.bus.disconnect(disable_torque)


class _ThreadedCamera:
    """Base class: a grab thread keeps `self._frame` fresh (BGR uint8)."""

    def __init__(self, name: str, flip: str | None, gain: float = 1.0):
        if flip not in _FLIP_CODES:
            raise ValueError(f"camera {name!r}: flip must be one of v/h/180/none, got {flip!r}")
        self.name = name
        self.flip_code = _FLIP_CODES[flip]
        # Software brightness. The RealSense under-exposes this dim room, and a
        # dark scene image measurably kills grasp planning (see README).
        self.gain = float(gain)
        self._frame: np.ndarray | None = None
        self._frame_t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{name}")

    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError

    def _loop(self) -> None:
        errors = 0
        last_log = 0.0
        while not self._stop.is_set():
            try:
                img = self._grab()
            except Exception as e:  # noqa: BLE001
                img = None
                errors += 1
                if time.monotonic() - last_log > 10.0:
                    print(f"[cam {self.name}] capture error x{errors}: {e}")
                    last_log = time.monotonic()
            if img is None:
                time.sleep(0.02)
                continue
            if self.flip_code is not None:
                img = cv2.flip(img, self.flip_code)
            if self.gain != 1.0:
                img = cv2.convertScaleAbs(img, alpha=self.gain, beta=0)
            with self._lock:
                self._frame = img
                self._frame_t = time.monotonic()

    def read(self, max_age_s: float = 1.0) -> np.ndarray | None:
        """Latest BGR frame, or None if nothing arrived in the last `max_age_s`."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._frame_t > max_age_s:
                return None
            return self._frame

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _set_v4l2(device: str, **ctrls: Any) -> None:
    """Best-effort v4l2-ctl call; silently skipped if v4l-utils is missing."""
    if not ctrls:
        return
    arg = ",".join(f"{k}={v}" for k, v in ctrls.items())
    try:
        subprocess.run(["v4l2-ctl", "-d", device, "-c", arg], check=False, capture_output=True)
    except FileNotFoundError:
        pass


class OpenCVCamera(_ThreadedCamera):
    """Any V4L2 webcam (wrist cams, Logitech C9xx, or a RealSense's RGB node).

    `device` may be an index (`4`), `/dev/video4`, or — recommended, because
    indices shuffle between boots — a `/dev/v4l/by-id/...-video-index0` path.
    """

    def __init__(self, name: str, device: str | int, width: int = 640, height: int = 480,
                 fps: int = 30, flip: str | None = None, fourcc: str | None = "MJPG",
                 white_balance_temperature: int | None = None,
                 v4l2_controls: dict[str, Any] | None = None, gain: float = 1.0):
        super().__init__(name, flip, gain)
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        dev_path = f"/dev/video{device}" if isinstance(device, int) else os.path.realpath(device)
        if not os.path.exists(dev_path):
            present = sorted(p for p in os.listdir("/dev") if p.startswith("video"))
            raise FileNotFoundError(
                f"camera {name!r}: {device} does not exist. Available: {present}. "
                "Run `v4l2-ctl --list-devices` or `uv run so101_client.py cameras`."
            )
        self.dev_path = dev_path
        cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"camera {name!r}: could not open {dev_path}")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._restore_auto_wb = white_balance_temperature is not None
        if white_balance_temperature is not None:
            # Auto white-balance drift between sessions can collapse the
            # predicted action deltas; pin it.
            _set_v4l2(dev_path, white_balance_automatic=0,
                      white_balance_temperature=int(white_balance_temperature))
        if v4l2_controls:
            _set_v4l2(dev_path, **v4l2_controls)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"[cam {name}] opened {dev_path} at {got[0]}x{got[1]}")
        self._thread.start()

    def _grab(self) -> np.ndarray | None:
        ok, img = self._cap.read()
        return img if ok else None

    def close(self) -> None:
        super().close()
        self._cap.release()
        if self._restore_auto_wb:
            _set_v4l2(self.dev_path, white_balance_automatic=1)


class RealSenseCamera(_ThreadedCamera):
    """Intel RealSense (D435/D455/...) colour stream via pyrealsense2. RGB only —
    MolmoAct2-SO100_101 has depth reasoning disabled."""

    def __init__(self, name: str, serial: str | None = None, width: int = 640,
                 height: int = 480, fps: int = 30, flip: str | None = None,
                 gain: float = 1.0, white_balance: int | None = None,
                 exposure: int | None = None):
        super().__init__(name, flip, gain)
        import pyrealsense2 as rs

        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError(
                f"camera {name!r}: no RealSense devices found. Needs a USB-3 data cable; "
                "check `rs-enumerate-devices`."
            )
        if serial is not None:
            device = next((d for d in devices
                           if d.get_info(rs.camera_info.serial_number) == str(serial)), None)
            if device is None:
                found = [d.get_info(rs.camera_info.serial_number) for d in devices]
                raise RuntimeError(f"camera {name!r}: RealSense serial {serial} not found (have {found})")
        else:
            device = devices[0]
        chosen = device.get_info(rs.camera_info.serial_number)
        usb = device.get_info(rs.camera_info.usb_type_descriptor)
        if not usb.startswith("3") and fps > 15:
            print(f"[cam {name}] RealSense is on USB {usb}; dropping to 15 fps")
            fps = 15
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(chosen)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipeline.start(cfg)
        self._pipeline = pipeline
        # Auto WB on a D435 under mixed desk light swings hard blue, which the
        # policy sees as a completely different scene. Pin it when asked.
        sensor = profile.get_device().first_color_sensor()
        for opt, val, auto in ((rs.option.white_balance, white_balance, rs.option.enable_auto_white_balance),
                               (rs.option.exposure, exposure, rs.option.enable_auto_exposure)):
            if val is None:
                continue
            try:
                sensor.set_option(auto, 0)
                sensor.set_option(opt, float(val))
            except Exception as e:  # noqa: BLE001
                print(f"[cam {name}] could not set {opt}: {e}")
        print(f"[cam {name}] RealSense {device.get_info(rs.camera_info.name)} "
              f"serial {chosen} (USB {usb}) at {width}x{height}@{fps}")
        self._thread.start()

    def _grab(self) -> np.ndarray | None:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        cf = frames.get_color_frame()
        return np.asanyarray(cf.get_data()).copy() if cf else None

    def close(self) -> None:
        super().close()
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


def make_camera(name: str, spec: dict[str, Any]) -> _ThreadedCamera:
    spec = dict(spec)
    kind = spec.pop("type", "opencv")
    if kind == "opencv":
        if "device" not in spec:
            raise ValueError(f"camera {name!r}: opencv cameras need a `device`")
        return OpenCVCamera(name, **spec)
    if kind == "realsense":
        return RealSenseCamera(name, **spec)
    raise ValueError(f"camera {name!r}: unknown type {kind!r} (expected opencv or realsense)")


def wait_for_frames(cameras: dict[str, _ThreadedCamera], timeout_s: float = 30.0) -> None:
    t0 = time.monotonic()
    pending = set(cameras)
    while pending and time.monotonic() - t0 < timeout_s:
        for n in list(pending):
            if cameras[n].read() is not None:
                print(f"[cam {n}] first frame after {time.monotonic() - t0:.1f}s")
                pending.discard(n)
        time.sleep(0.1)
    if pending:
        raise RuntimeError(f"cameras produced no frames within {timeout_s:.0f}s: {sorted(pending)}")

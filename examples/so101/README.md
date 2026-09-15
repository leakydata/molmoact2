# MolmoAct2 on SO-100/101 (one or two arms)

Zero-shot control of SO-101 follower arms with [`allenai/MolmoAct2-SO100_101`](https://huggingface.co/allenai/MolmoAct2-SO100_101). Adapted from [irenegracekp/molmoact2-so101](https://github.com/irenegracekp/molmoact2-so101), and split into the same server/client shape as the DROID and YAM examples:

```
 GPU box                                   robot box (can be the same machine)
 ┌──────────────────────────────┐   HTTP   ┌──────────────────────────────────────────────┐
 │ host_server_so101.py  :8101  │ ◄──────► │ so101_client.py                              │
 │ model, bf16, CUDA graphs     │  /act    │ LeRobot SOFollower ×N, cameras, safety,      │
 │ (repo-root uv env)           │          │ temporal ensembling (examples/so101 uv env)  │
 └──────────────────────────────┘          └──────────────────────────────────────────────┘
```

They use two separate environments because LeRobot needs `huggingface-hub>=1.0` and the server's `transformers 4.57` needs `<1.0`.

| File | What |
| --- | --- |
| `host_server_so101.py` | FastAPI server. 2 cameras (`scene_cam`, `wrist_cam`), state `(6,)`, `norm_tag="so100_so101_molmoact2"`, port 8101. |
| `so101_client.py` | `cameras` / `check` / `run` subcommands. |
| `so101_hardware.py` | Threaded LeRobot arm driver, OpenCV + RealSense cameras. |
| `so101_runtime.py` | HTTP client, async producer, per-arm temporal-ensembling consumers. |
| `configs/single_arm.yaml`, `configs/two_arms.yaml` | Hardware wiring and safety limits. |

## Hardware

* SO-101 (or SO-100) follower arm(s), calibrated with LeRobot.
* A **third-person scene camera**: RealSense D435/D455 (colour stream only), or any USB webcam.
* A **wrist camera** per arm: the standard SO-101 wrist USB cam.
* GPU for the server: bf16 needs about 13.5 GB with CUDA graphs (measured on an RTX 4090).

## 1. Server

From the repo root:

```bash
uv sync
uv run hf download allenai/MolmoAct2-SO100_101      # ~21 GB
./run_so101.sh                                     # bf16 + CUDA graphs on :8101
curl http://127.0.0.1:8101/act                     # health, includes training state ranges
```

On a machine with several GPUs, pin the card by PCI order. CUDA's default numbering is fastest-first, which differs from `nvidia-smi`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ./run_so101.sh
```

Measured on an RTX 4090 with bf16:

| | per `/act` call |
| --- | --- |
| no CUDA graphs | ~1.2 s (longer than a 1 s chunk, so motion stutters) |
| `--cuda-graph` (default in `run_so101.sh`) | ~210–270 ms |
| first call per new prompt length | ~2.5 s (graph capture, one-time) |

Upstream caches only one action-expert CUDA graph, and the graph is keyed on the prompt's token count. With two arms using different prompts, every request would re-capture. The server keeps an LRU of graphs instead (`--cuda-graph-cache`, default 2).

## 2. Client environment

```bash
cd examples/so101
uv sync          # Python 3.12, lerobot[feetech] 0.6.x, pyrealsense2, CPU-only torch
```

Everything below runs from `examples/so101/`.

## 3. Calibrate each arm (LeRobot)

```bash
uv run lerobot-find-port
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=left_follower
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=right_follower
```

The `--robot.id` you choose is the `calibration_id` in the config. It maps to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`. Serial ports can swap between reboots, so prefer the `/dev/serial/by-id/...` paths.

## 4. Cameras

```bash
uv run so101_client.py cameras                                   # by-id paths + RealSense serials
uv run so101_client.py cameras --config configs/single_arm.yaml --snapshot snaps/   # or --show
```

* Put the `/dev/v4l/by-id/...-video-index0` path in `device:`. `/dev/videoN` numbers change between boots.
* The number embedded in a RealSense's by-id path is **not** its serial. Use the `serial=` value printed above.
* **Webcam instead of the RealSense:** change the scene camera to `type: opencv` and give it a `device:`. Nothing else changes.
* Set the wrist cam's `flip:` so the image is upright as mounted (Irene's rig needed `180`).

## 5. Check the joint convention (do this before the arm moves)

MolmoAct2 was trained on data in the old LeRobot (v2.1) joint convention. Current LeRobot calibrations use v3.0. The config converts between them with `model = signs * arm + offsets`, with defaults `signs=[1,-1,1,1,1,1]` and `offsets=[0,90,90,0,0,0]` ([LeRobot backward-compat doc](https://huggingface.co/docs/lerobot/backwardcomp)).

```bash
uv run so101_client.py check --config configs/single_arm.yaml            # torque on, holds pose
uv run so101_client.py check --config configs/single_arm.yaml --limp     # torque off: support the arm!
```

In the folded rest pose, the "model frame" column should read roughly `[0, 189, 181, 60, -4, 1]`, which is the model card's sample state. Move the arm around the workspace. A joint flagged `OUT OF RANGE` that stays out, or moves the wrong way, means its sign or offset is wrong.

The **gripper** is the least certain joint. LeRobot reports it on a 0–100 scale, while the training data sits around 0–44. If the gripper never closes or opens properly, that scale is the first thing to look at.

## 6. Run

```bash
# 1) model only, arms hold still: sanity-check the printed a0-state deltas
uv run so101_client.py run --config configs/single_arm.yaml --prompt "pick up the lemon and put it in the bowl" --dry-run --show

# 2) for real
uv run so101_client.py run --config configs/single_arm.yaml --prompt "pick up the lemon and put it in the bowl"

# two arms, one prompt each
uv run so101_client.py run --config configs/two_arms.yaml \
    --arm-prompt left="pick up the red block" --arm-prompt right="put the cup on the plate"
```

Stop with Ctrl+C. By default torque is released on exit and the arm drops, so keep it over something soft or near rest. Set `hold_torque_on_exit: true` to keep it stiff instead.

`--simulate` replaces the arms with fake followers, which lets you test cameras, server and timing without any hardware connected.

### Safety layers (per arm, in `configs/*.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `max_chunk_jump_deg` | 30 | Drops a chunk if its first action is further than this from the current pose. A wrong joint convention trips this before the arm moves. |
| `max_step_deg` | 5 | Largest joint change per control tick (30 Hz ⇒ ≤150°/s); the whole vector is scaled. |
| `joint_min` / `joint_max` | none | Hard clamps in LeRobot degrees. |
| (internal) | 4° | Rate limit per servo write in the bus thread. |

## Two arms: what to expect

* **The checkpoint is single-arm** (`setup_type: single so100/so101 robotic arm`). Each arm is queried on its own with its own state, wrist image and prompt, so the arms don't coordinate and don't avoid each other. Give them separate workspaces. A real bimanual SO-101 policy needs fine-tuning (see the LeRobot MolmoAct2 docs).
* The producer alternates arms, so each arm gets a new chunk every ~0.5 s. If the GPU is shared with other work, a slow cycle can outlast the 1 s chunk. The arm then logs `holding position` and waits, which is safe but jerky.
* Both arms can share one scene camera, or each can point `scene_camera` at its own.

## Known limitations

* The wrist view is somewhat out of distribution: the checkpoint's sample inputs are two third-person RealSense views. If behaviour is poor, try `scene_only: true`, which sends the scene image twice.
* No depth is used (`enable_depth_reasoning=False`; this checkpoint raises if you enable it).
* Performance outside simple pick-and-place-style tasks isn't guaranteed.

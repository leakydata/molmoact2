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
| `so101_client.py` | `cameras` / `check` / `probe` / `selftest` / `demo` / `run` subcommands. |
| `so101_hardware.py` | Threaded LeRobot arm driver, OpenCV + RealSense cameras. |
| `so101_runtime.py` | HTTP client, async producer, per-arm consumers (`ensemble` or full-chunk `commit` execution). |
| `so101_preview.py` | Browser camera view + status at `http://127.0.0.1:8102/` (`--show`); LeRobot pins headless OpenCV, so `cv2.imshow` is not available. |
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

On a machine with several GPUs, pin the card **by UUID** — CUDA's default numbering
is fastest-first, `nvidia-smi` numbers by PCI order, and a shell may already export
`CUDA_VISIBLE_DEVICES`, so an index can silently select the wrong card:

```bash
nvidia-smi -L                                   # GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-...)
SO101_GPU=GPU-<uuid> ./run_so101.sh             # run_so101.sh pins this for you
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
uv sync          # Python 3.12, lerobot[feetech] 0.6.x, pyrealsense2, CUDA torch
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
| `max_chunk_jump_deg` | 60 | Drops a chunk if its first action is further than this from the current pose. A wrong joint convention trips this before the arm moves; start low (30) until `check` looks right. |
| `max_step_deg` | 15 | Largest joint change per control tick (30 Hz ⇒ ≤450°/s); the whole vector is scaled. |
| `joint_min` / `joint_max` | none | Hard clamps in LeRobot degrees. |
| (internal) | 4° | Rate limit per servo write in the bus thread. |

`best_of: N` (or `--best-of N`) draws N plans per cycle and keeps the one with
the most gripper travel. It selects among the policy's own proposals rather than
authoring motion, which helps when most samples are "approach with open jaws",
but it multiplies inference latency (N × ~200 ms) and a chunk only covers 1 s —
past N≈3 the chunk expires before it can be executed. Default 1.

## Making the policy actually move (lessons from real runs)

Zero-shot MolmoAct2 is picky about its *inputs*, not about this client. On a real
SO-101 the same checkpoint swings between "reaches out, grasps, places the object"
and "plans 2° and parks". What decided it, in order of impact:

0. **The scene camera must look *down* at the table.** This dominates everything
   else below. Measured on one rig by replaying saved frames through the server
   30× per condition and counting how many plans contain any gripper travel:

   | scene camera | P(plan closes the gripper) |
   | --- | --- |
   | elevated, angled down at the tabletop | **50–75%** |
   | same rig, camera knocked down to table height, looking across | **0–3%** |

   The edge-on view is not recoverable in software: cropping to the table,
   sending the scene twice, the wrist twice, brightening, CLAHE, every prompt
   wording, and flow-matching steps 4/10/20/32 all stayed at 0–4%. Raising the
   camera fixed it. If the policy "goes dumb" after someone bumps a camera,
   check the camera geometry before touching anything else.

   A dark or colour-cast scene image costs almost as much. Pin the RealSense's
   white balance and exposure (`white_balance:`, `exposure:`, `gain:` in the
   camera config) — its auto WB swings hard blue under mixed desk light, and
   the policy sees a blue room as a different scene entirely.

   Careful with this measurement: reach magnitude alone is *not* task behaviour
   (the checkpoint always drifts back toward its training range), and gripper
   travel is only meaningful once the gripper is near the object — a plan that
   keeps the jaws open while still far away is correct. Score P(grasp) over many
   samples, never a mean over a handful: the policy is stochastic and a single
   lucky draw looks like a 27° "win" that vanishes on the next frame.

1. **Start from a mid-range pose, never the folded rest pose.** In the fold, the
   elbow/shoulder sit past the checkpoint's `q99`, and every plan is just a small
   pull back toward the training range — no task behaviour at all. Pass
   `--start-pose 3.1,124.5,122.8,57.8,-11.1,4.9` (the training median, model frame)
   and the same scene suddenly produces 20–70° reaches with the gripper opening.
   Sweeping candidate poses offline (`probe`-style, no hardware) found
   `-20,124.5,110,40,-11.1,4.9` ~27% better again on one rig. Two traps when
   scoring poses: a pose outside the median scores high merely because the plan
   drives *back* toward the training range (no gripper motion — not task
   behaviour), and starting with the gripper already open removes the grasp
   entirely (gripper motion collapses from ~16° to ~1°).
2. **Re-home whenever it stalls.** After finishing (or wandering into a pose it
   dislikes) the policy parks and plans nothing. `--rehome-after SECONDS` +
   `--stall-deg DEG` ramp back to the start pose — the run then cycles
   "reset → attempt → reset" instead of freezing. Queued chunks are dropped on
   re-home; they were planned for the pose being left.
3. **Prompt wording matters more than expected.** Short, concrete, and naming an
   object that is actually visible: `pick up the marker and put it in the mug`.
   Naming something out of frame ("the black bin" when no bin is in view) makes it
   retract to rest. `so101_client.py probe` scores wordings live without moving
   the arm.
4. **Don't switch prompts mid-carry.** Changing the instruction while the gripper
   holds something makes it re-plan from scratch and drop the object. Use one
   full-task prompt.
5. **Object placement and contrast.** Objects want to be in the open area in front
   of the arm (the preview draws a `place object here` box), not at its base or the
   frame edge. A black marker on dark wood is near-invisible to the scene camera;
   bright, chunky objects work best.
6. **Per-arm wrist offsets may be needed.** LeRobot's calibration zero is whatever
   pose you held at calibration time, so `joint_offsets` sometimes needs per-joint
   corrections beyond the documented v3.0→v2.1 conversion. With the wrist roll
   wrong by ~78° the policy retracted instead of reaching. Check with
   `so101_client.py check`, then A/B the offsets with `probe`.

## Two arms: what to expect

* **The checkpoint is single-arm** (`setup_type: single so100/so101 robotic arm`). Each arm is queried on its own with its own state, wrist image and prompt, so the arms don't coordinate and don't avoid each other. Give them separate workspaces. A real bimanual SO-101 policy needs fine-tuning (see the LeRobot MolmoAct2 docs).
* The producer alternates arms, so each arm gets a new chunk every ~0.5 s. If the GPU is shared with other work, a slow cycle can outlast the 1 s chunk. The arm then logs `holding position` and waits, which is safe but jerky.
* Both arms can share one scene camera, or each can point `scene_camera` at its own.

## Known limitations

* The wrist view is somewhat out of distribution: the checkpoint's sample inputs are two third-person RealSense views. If behaviour is poor, try `scene_only: true`, which sends the scene image twice.
* No depth is used (`enable_depth_reasoning=False`; this checkpoint raises if you enable it).
* Performance outside simple pick-and-place-style tasks isn't guaranteed.

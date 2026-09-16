export HF_HUB_ENABLE_HF_TRANSFER=1
# Pin the RTX 4090 by UUID so the result doesn't depend on CUDA's device
# ordering or an inherited CUDA_VISIBLE_DEVICES (the GTX 1080 must never be
# used). Override with SO101_GPU=<uuid or index> on other machines.
export CUDA_VISIBLE_DEVICES="${SO101_GPU:-GPU-9a77d842-d580-b4c0-8552-58404d12d924}"
# --cuda-graph: ~250 ms/call instead of ~1.2 s on an RTX 4090 (chunks cover only 1 s).
uv run python examples/so101/host_server_so101.py --host 0.0.0.0 --port 8101 --dtype bfloat16 --cuda-graph "$@"

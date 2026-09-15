export HF_HUB_ENABLE_HF_TRANSFER=1
# --cuda-graph: ~250 ms/call instead of ~1.2 s on an RTX 4090 (chunks cover only 1 s).
uv run python examples/so101/host_server_so101.py --host 0.0.0.0 --port 8101 --dtype bfloat16 --cuda-graph "$@"

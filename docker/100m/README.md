# 100M Image Contents

This lightweight image is only a bootstrap layer for `TYPE=100M bash dep.sh`.
It intentionally includes only:

- `python:3.11-slim-bookworm`
- `bash`
- `ca-certificates`
- `git`
- `libvulkan1`
- `libegl1`, `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`

It intentionally excludes heavyweight runtime components such as:

- `torch`, `torchvision`, `torchaudio`
- `mani_skill`, `sapien`
- checkpoints and datasets
- optional heavy packages such as `flash_attn` and `deepspeed`

When `TYPE=100M` is used, `dep.sh` starts the container from this image and then
runs `dep-non-docker.sh` inside the container to install the remaining Python runtime.

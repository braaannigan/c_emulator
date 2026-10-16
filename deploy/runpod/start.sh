#!/usr/bin/env bash
# Worker start command for the runpod/pytorch base image: install the few
# pipeline dependencies the image lacks, fetch the handler from main and serve.
set -euo pipefail
PIP_DEPS="runpod xarray netcdf4 pyyaml scipy matplotlib"
# A venv layered on the system site-packages keeps the image's torch but lets
# pip upgrade packages (e.g. cryptography) that Debian installed system-wide.
python -m venv --system-site-packages /opt/venv
/opt/venv/bin/pip install --quiet --no-cache-dir $PIP_DEPS
export PATH=/opt/venv/bin:$PATH
rm -rf /opt/c_emulator
git clone --quiet --depth 1 https://github.com/braaannigan/c_emulator.git /opt/c_emulator
cd /opt/c_emulator
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
exec env PYTHONPATH=/opt/c_emulator python -u deploy/runpod/handler.py

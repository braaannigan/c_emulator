#!/usr/bin/env bash
# Worker start command for the runpod/pytorch base image: install the few
# pipeline dependencies the image lacks, fetch the handler from main and serve.
set -euo pipefail
PIP_DEPS="runpod xarray netcdf4 pyyaml scipy"
pip install --quiet --no-cache-dir $PIP_DEPS 2>/dev/null || pip install --quiet --no-cache-dir --break-system-packages $PIP_DEPS
rm -rf /opt/c_emulator
git clone --quiet --depth 1 https://github.com/braaannigan/c_emulator.git /opt/c_emulator
cd /opt/c_emulator
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
exec env PYTHONPATH=/opt/c_emulator python -u deploy/runpod/handler.py

#! /bin/bash

# This is a script for initializing the environment for RUNPOD RTX 4090

set -e

cd /workspace/
git clone https://github.com/yangjiwoong1/HAT_RCF.git

# AID dataset download
(
    pip install gdown
    apt-get update && apt-get install -y unzip
    cd HAT_RCF/datasets/
    gdown 1d_Wq_U8DW-dOC3etvF4bbbWMOEqtZwF7 -O AID-dataset.zip
    unzip AID-dataset.zip
    cd .. && python tools/remove_npy_from_folder.py
) & # background process

# config virtual environment
(
    python -m venv --system-site-packages venv-system
    source venv-system/bin/activate
    cd HAT_RCF
    pip install -r requirements.txt
    python setup.py develop
) &

wait

# download pretrained model
cd /workspace/HAT_RCF/pretrained_models/ # RCF weights (required)
gdown 1oxlHQCM4mm5zhHzmE7yho_oToU5Ucckk

cd /workspace/HAT_RCF/experiments/pretrained_models/ # HAT weights (optional)
gdown 1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI

### manual steps ###
# 1. /workspace/venv-system/lib/python3.11/site-packages/basicsr/data/degradations.py로 이동해서
# 2. "from torchvision.transforms.functional_tensor import rgb_to_grayscale" -> "from torchvision.transforms.functional import rgb_to_grayscale" (_tensor 삭제)
#! /bin/bash
set -e
wandb login # TODO: Does it need to be run?
python hat/train.py -opt options/train/HATRCF/train_HATRCF_SRx4_from_scratch.yml --launcher none

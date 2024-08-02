#!/bin/bash

python3 train.py --multirun wandb.project="GFN-diffusion reproduce" experiment=GFN/OffPolicy/TB+Expl+LP+LS,GFN/OffPolicy/TB+Expl+LS "$*"
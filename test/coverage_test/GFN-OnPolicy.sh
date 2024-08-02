#!/bin/bash

python3 train.py --multirun wandb.project="GFN-diffusion reproduce" experiment=GFN/OnPolicy/TB,GFN/OnPolicy/TB+Expl,GFN/OnPolicy/TB+Expl+LP "$*"
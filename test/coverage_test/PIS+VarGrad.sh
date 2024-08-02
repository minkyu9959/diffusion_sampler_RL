#!/bin/bash

python3 train.py --multirun wandb.project="GFN-diffusion reproduce" experiment=GFN/OnPolicy/VarGrad+Expl,PIS/PIS+LP "$*"
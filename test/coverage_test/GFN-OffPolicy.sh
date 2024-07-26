#!/bin/bash

python3 train.py --multirun wandb.project="DEBUG" experiment=GFN/TB+Expl+LP+LS,GFN/TB+Expl+LS "$*"
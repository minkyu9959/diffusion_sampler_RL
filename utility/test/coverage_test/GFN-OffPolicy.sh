#!/bin/bash

python3 train.py --multirun experiment=GFN/OffPolicy/TB+Expl+LP+LS,GFN/OffPolicy/TB+Expl+LS "$*"
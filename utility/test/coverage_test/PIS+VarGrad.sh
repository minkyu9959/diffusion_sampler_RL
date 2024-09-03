#!/bin/bash

python3 train.py --multirun experiment=GFN/OnPolicy/VarGrad+Expl,PIS/PIS+LP "$*"
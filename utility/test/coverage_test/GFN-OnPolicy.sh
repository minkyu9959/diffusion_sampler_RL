#!/bin/bash

python3 train.py --multirun experiment=GFN/OnPolicy/TB,GFN/OnPolicy/TB+Expl,GFN/OnPolicy/TB+Expl+LP "$*"
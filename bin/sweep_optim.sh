#!/bin/bash

set -Eeuo pipefail

tmux_session="ManyWell512"
GPUS=(1 2 3 6)

H="4096"
S="2048"
T="512"

LR_POLICY="1e-4"
WEIGHT_DECAY="true false"
TRAJ="100 200"

group="baseline"

epoch=30000

count=0

for weight_decay in ${WEIGHT_DECAY[@]}; do
    for TRAJ in ${TRAJ[@]}; do
        gpu_id=${GPUS[$count]}

        experiment="PIS/PIS+LP"
        energy="ManyWell512"

        model_cfg="\
            model.forward_model.hidden_dim=$H \
            model.state_encoder.s_emb_dim=$S \
            model.time_encoder.t_emb_dim=$T \
            model.langevin_scaler.out_dim=512 \
            model.trajectory_length=$TRAJ \
            "
        model_cfg=$(echo "$model_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        optim_cfg="\
            model.optimizer_cfg.lr_policy=$LR_POLICY\
            model.optimizer_cfg.use_weight_decay=$weight_decay \
            "
        optim_cfg=$(echo "$optim_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        ARGS="experiment=$experiment energy=$energy $model_cfg $optim_cfg +group_tag=$group train.epochs=$epoch"

        tmux new-window -t $tmux_session: 
        tmux send-keys -t $tmux_session: "CUDA_VISIBLE_DEVICES=${gpu_id} python3 sampler/train.py ${ARGS}" C-m

        count=$(( count+1 ))
    done
done

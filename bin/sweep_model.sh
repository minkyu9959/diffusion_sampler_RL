#!/bin/bash

set -Eeuo pipefail

tmux_session="ManyWell"
GPUS=(0 1)

H="256"
S="128"
T="128"

group="CMCD"
epoch=4000

BASE_STD="4.0"
pstd="1.0"
TRAJECTORY_LENGTH="512 1024"
BATCH_SIZE="300"

LR_POLICY="1e-3"

count=0

for base_std in ${BASE_STD[@]}; do
    for trajectory_length in ${TRAJECTORY_LENGTH[@]}; do
        gpu_id=${GPUS[$count]}

        experiment="AnnealedDB/OnPolicy/CMCD"
        energy="ManyWell"

        model_cfg="\
            model.control_model.hidden_dim=$H \
            model.state_encoder.s_emb_dim=$S \
            model.time_encoder.t_emb_dim=$T \
            model.base_std=$base_std \
            model.trajectory_length=$trajectory_length \
            model.prior_energy.std=$pstd \
            model.clipping=false \
            "

        model_cfg=$(echo "$model_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        optim_cfg="\
            model.optimizer_cfg.lr_policy=$LR_POLICY \
            model.optimizer_cfg.use_weight_decay=true \
            "
        optim_cfg=$(echo "$optim_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        ARGS="experiment=$experiment energy=$energy train.fwd_loss=tb-avg $model_cfg $optim_cfg +group_tag=$group train.epochs=$epoch train.batch_size=$BATCH_SIZE"

        tmux new-window -t $tmux_session: 
        tmux send-keys -t $tmux_session: "CUDA_VISIBLE_DEVICES=${gpu_id} python3 sampler/train.py ${ARGS}" C-m

        count=$(( count+1 ))
    done
done


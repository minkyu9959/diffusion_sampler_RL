#!/bin/bash

set -Eeuo pipefail

tmux_session="ManyWell512"
GPUS=(6 7)

H="1024"
S="512"
T="512"

group="baseline"
epoch=20000

BASE_STD="4.0"
TRAJECTORY_LENGTH="256 512"
BATCH_SIZE="300"

LR_POLICY="1e-4"

count=0

for t_scale in ${BASE_STD[@]}; do
    for trajectory_length in ${TRAJECTORY_LENGTH[@]}; do
        gpu_id=${GPUS[$count]}

        experiment="GFN/OffPolicy/TB+Expl+LP+LS"
        energy="ManyWell512"

        model_cfg="\
            model.forward_policy.hidden_dim=$H \
            model.state_encoder.s_emb_dim=$S \
            model.time_encoder.t_emb_dim=$T \
            model.t_scale=$t_scale \
            model.trajectory_length=$trajectory_length \
            model.langevin_scaler.out_dim=512 \
            "

        model_cfg=$(echo "$model_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        optim_cfg="\
            model.optimizer_cfg.lr_policy=$LR_POLICY \
            model.optimizer_cfg.use_weight_decay=true \
            "
        optim_cfg=$(echo "$optim_cfg" | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')

        ARGS="experiment=$experiment energy=$energy $model_cfg $optim_cfg +group_tag=$group train.epochs=$epoch train.batch_size=$BATCH_SIZE eval.eval_data_size=1000 eval.plot_sample_size=1000"

        tmux new-window -t $tmux_session: 
        tmux send-keys -t $tmux_session: "CUDA_VISIBLE_DEVICES=${gpu_id} python3 sampler/train.py ${ARGS}" C-m

        count=$(( count+1 ))
    done
done

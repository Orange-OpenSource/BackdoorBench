#!/bin/bash

# Software Name: BackdoorBench/run_attack
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Run an attack according to the given args

export MPLBACKEND=AGG
set -e

#vbd_utils/run_attack.sh single tiny trigger_green_square 0.2 0_all 0
#vbd_utils/run_attack.sh single tiny trigger_color_grid 0.2 0_all 0

#vbd_utils/run_attack.sh multi tiny square 0.2 trigger_green_square 3 0,255,0 100_1 100 1 100 0
#vbd_utils/run_attack.sh multi tiny square 0.2 trigger_color_grid 3 255,255,255 1_1 0 1 1 0


attack_type=$1

if [[ "$attack_type" == "single" ]]; then
    #dataset=cifar10 or tiny
    #patt=trigger_green_square
    #blend_rate=0.2
    #mode=0_all

    dataset=$2
    patt=$3
    blend_rate=$4
    mode=$5
    target=`cut -d'_' -f1 <<<"${mode}"`
    seed=$6

    if [[ "$dataset" == "cifar10" ]]; then
        bsize=""
    elif [[ "$dataset" == "tiny" ]]; then
        bsize="--batch_size 64"
    else
        echo "unknown dataset ${dataset}"
        exit 1
    fi

    python ./attack/maskblended.py --yaml_path config/attack/prototype/${dataset}.yaml --attack_trigger_img_path resource/maskblended/${dataset}/${patt}.png  --attack_trigger_mask_path resource/maskblended/${dataset}/${patt}_mask.png --attack_train_blended_alpha ${blend_rate} --attack_test_blended_alpha ${blend_rate} --save_folder_name ${dataset}_maskblended_${mode}_${patt}_${blend_rate}_${seed} --attack_target ${target} --random_seed ${seed} ${bsize}

elif [[ "$attack_type" == "multi" ]]; then

    #dataset=cifar10 or tiny
    #pat_type=square   square or grid or multi_color_grid
    #blend_rate=0.2
    #pat=trigger_green_square
    #pat_size=3
    #color=0,255,0
    #mode=100_1
    #start_label=100
    #group_labels=1
    #shift_label=100
    #seed=0

    dataset=$2
    pat_type=$3
    blend_rate=$4
    pat=$5
    pat_size=$6
    color=$7
    mode=$8
    start_label=$9
    group_labels=${10}
    shift_label=${11}
    seed=${12}

    if [[ "$dataset" == "cifar10" ]]; then
        bsize=""
        num_labels=10
        w_h=32
    elif [[ "$dataset" == "tiny" ]]; then
        bsize="--batch_size 64"
        num_labels=200
        w_h=64
    else
        echo "unknown dataset ${dataset}"
        exit 1
    fi

    tmp_dir=`mktemp -d`
    python vbd_utils/generate_pattern.py --output ${tmp_dir} --width ${w_h} --height ${w_h} --size ${pat_size} --type ${pat_type} --color ${color} --num_labels ${num_labels} --start_label ${start_label} --group_labels ${group_labels} --shift_label ${shift_label}
    python ./attack/maskblended.py --attack_label_trans multitarget --yaml_path config/attack/prototype/${dataset}.yaml --attack_trigger_img_path ${tmp_dir}  --attack_train_blended_alpha ${blend_rate} --attack_test_blended_alpha ${blend_rate} --save_folder_name ${dataset}_maskblended_${mode}_${pat}_${blend_rate}_${seed} --random_seed ${seed} ${bsize}
    rm -rf ${tmp_dir}

else
    echo "Unknown attack type ${attack_type}"
    exit 1
fi


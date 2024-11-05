#!/usr/bin/env bash

# Software Name: BackdoorBench/run_all_attacks
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Run all the attacks planned for the evaluation

set -e

if [[ "${1}" == "dryrun" ]]; then
    CMD=echo
else
    CMD=""
fi

function run_attack () {
    attack_type=$1
    if [[ "$attack_type" == "single" ]]; then
        dataset=$2
        pat=$3
        blend_rate=$4
        mode=$5
        seed=$6
    elif [[ "$attack_type" == "multi" ]]; then
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
    else
        echo "unknown attack type: ${attack_type}" >&2
        exit 1
    fi

    dir_name=${dataset}_maskblended_${mode}_${pat}_${blend_rate}_${seed}
    #if [ -f "record/${dir_name}/attack_result.pt" ]; then
    if [ -d "record/${dir_name}" ]; then
        echo "record/${dir_name} already exists (ignored)" >&2
    else
		if [[ "${CMD}" == "" ]]; then
			echo "Running vbd_utils/run_attack.sh $@ ..."
		fi
        ${CMD} vbd_utils/run_attack.sh $@
    fi
} 


#for single
for dataset in cifar10 tiny
do
    for seed in 0 1 2 3 4
    do
        mode=${seed}_all
        for patt in small_hello_kitty big_hello_kitty trigger_color_grid trigger_green_square trigger_white_square trigger_white_grid
        do
            for blend_rate in 0.1 0.2 0.5 1.0
            do
                run_attack single ${dataset} ${patt} ${blend_rate} ${mode} ${seed}
            done
        done
    done
done

#for multi
extended=False
for blend_rate in 0.1 0.2 0.5 1.0
do
    for seed in 0 1 2 3 4
    do
        run_attack multi cifar10 square ${blend_rate} trigger_green_square 3 0,255,0 1_1 0 1 1 ${seed}
        run_attack multi cifar10 square ${blend_rate} trigger_white_square 3 255,255,255 1_1 0 1 1 ${seed}
        run_attack multi cifar10 grid ${blend_rate} trigger_white_grid 3 255,255,255 1_1 0 1 1 ${seed}
        run_attack multi cifar10 multi_color_grid ${blend_rate} trigger_color_grid 3 0,255,0 1_1 0 1 1 ${seed}
        
        run_attack multi tiny square ${blend_rate} trigger_green_square 3 0,255,0 1_1 0 1 1 ${seed}
        run_attack multi tiny square ${blend_rate} trigger_white_square 3 255,255,255 1_1 0 1 1 ${seed}
        run_attack multi tiny grid ${blend_rate} trigger_white_grid 3 255,255,255 1_1 0 1 1 ${seed}
        run_attack multi tiny multi_color_grid ${blend_rate} trigger_color_grid 3 0,255,0 1_1 0 1 1 ${seed}
        
        if [[ "$extended" == "True" ]]; then
            run_attack multi cifar10 square ${blend_rate} trigger_green_square 3 0,255,0 5_1 5 1 5 ${seed}
            run_attack multi cifar10 square ${blend_rate} trigger_white_square 3 255,255,255 5_1 5 1 5 ${seed}
            run_attack multi cifar10 grid ${blend_rate} trigger_white_grid 3 255,255,255 5_1 5 1 5 ${seed}
            run_attack multi cifar10 multi_color_grid ${blend_rate} trigger_color_grid 3 0,255,0 5_1 5 1 5 ${seed}
            
            for gsize in 3 6
            do
                if [[ "$gsize" == "3" ]]; then
                    gs_pref=""
                else
                    gs_pref="s${gsize}"
                fi
                
                run_attack multi tiny square ${blend_rate} trigger_green_square ${gsize} 0,255,0 g20${gs_pref}_1 0 20 1 ${seed}
                run_attack multi tiny square ${blend_rate} trigger_white_square ${gsize} 255,255,255 g20${gs_pref}_1 0 20 1 ${seed}
                run_attack multi tiny grid ${blend_rate} trigger_white_grid ${gsize} 255,255,255 g20${gs_pref}_1 0 20 1 ${seed}
                run_attack multi tiny multi_color_grid ${blend_rate} trigger_color_grid ${gsize} 0,255,0 g20${gs_pref}_1 0 20 1 ${seed}
                
                run_attack multi tiny square ${blend_rate} trigger_green_square ${gsize} 0,255,0 g10${gs_pref}_1 0 10 1 ${seed}
                run_attack multi tiny square ${blend_rate} trigger_white_square ${gsize} 255,255,255 g10${gs_pref}_1 0 10 1 ${seed}
                run_attack multi tiny grid ${blend_rate} trigger_white_grid ${gsize} 255,255,255 g10${gs_pref}_1 0 10 1 ${seed}
                run_attack multi tiny multi_color_grid ${blend_rate} trigger_color_grid ${gsize} 0,255,0 g10${gs_pref}_1 0 10 1 ${seed}
                
                run_attack multi tiny square ${blend_rate} trigger_green_square ${gsize} 0,255,0 100${gs_pref}_1 100 1 100 ${seed}
                run_attack multi tiny square ${blend_rate} trigger_white_square ${gsize} 255,255,255 100${gs_pref}_1 100 1 100 ${seed}
                run_attack multi tiny grid ${blend_rate} trigger_white_grid ${gsize} 255,255,255 100${gs_pref}_1 100 1 100 ${seed}
                run_attack multi tiny multi_color_grid ${blend_rate} trigger_color_grid ${gsize} 0,255,0 100${gs_pref}_1 100 1 100 ${seed}
            done
        fi
    done
done



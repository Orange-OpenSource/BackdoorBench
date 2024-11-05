#!/bin/bash

# Software Name: BackdoorBench/run_defense
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Run a defense according to the given args

set -e
set -x

export MPLBACKEND=AGG

if [[ "$1" == "" ]]; then
    echo "Usage: defense_type dir_name"
    exit 1
fi

def_name=$1
dir_name=$2

dataset=`cut -d'_' -f1 <<<"${dir_name}"`
seed=`echo "${dir_name}" | awk -F '_' '{print $(NF)}'`


if [ ! -f "record/${dir_name}/attack_result.pt" ]; then
    echo "record/${dir_name}/attack_result.pt does not exist"
    exit 1
fi


if [[ "$dataset" == "cifar10" ]]; then
    bsize=""
    n_clean_sample=100
    #tlayer="layer4.0.conv2"
    tlayer="layer4"
    senti_tlayer="layer4.1.conv2"
elif [[ "$dataset" == "tiny" ]]; then
    bsize="--batch_size 64"
    n_clean_sample=2000
    #tlayer="layer4.0.conv1"
    tlayer="layer4"
    senti_tlayer="layer4.1.conv2"
else
    bsize=""
    n_clean_sample=100
    #tlayer="layer4.0.conv1"
    tlayer="layer4"
    senti_tlayer="layer4.1.conv2"
fi

if [[ "$def_name" == "abl" ]]; then
    if [ -f "record/${dir_name}/detection_pretrain/abl/detection_info.csv" ]; then
        echo "record/${dir_name}/detection_pretrain/abl/detection_info.csv" exists
    else
        python ./detection_pretrain/abl.py --result_file ${dir_name} --yaml_path ./config/detection/abl/${dataset}.yaml --dataset ${dataset} --csv_save_path record/${dir_name}/detection_pretrain/abl/detec_results.csv --random_seed ${seed} ${bsize}
    fi
elif [[ "$def_name" == "agpd" ]]; then
    if [ -f "record/${dir_name}/detection_pretrain/agpd/results_${n_clean_sample}.csv" ]; then
        echo "record/${dir_name}/detection_pretrain/agpd/results_${n_clean_sample}.csv" exists
    else
        python ./detection_pretrain/agpd.py --result_file ${dir_name} --yaml_path ./config/detection/agpd/${dataset}.yaml --dataset ${dataset} --tau 0.05 --clean_sample_num ${n_clean_sample} --csv_save_path record/${dir_name}/detection_pretrain/agpd/results_${n_clean_sample}.csv --random_seed ${seed} ${bsize}
    fi
elif [[ "$def_name" == "scan" ]]; then
    if [ -f "record/${dir_name}/detection/scan_pretrain/detection_info_${n_clean_sample}.csv" ]; then
        echo "record/${dir_name}/detection/scan_pretrain/detection_info_${n_clean_sample}.csv" exists
    else
        python ./detection_pretrain/scan.py --result_file ${dir_name} --yaml_path ./config/detection/scan/${dataset}.yaml --dataset ${dataset} --clean_sample_num ${n_clean_sample} --target_layer ${tlayer} --random_seed ${seed} ${bsize}
        mv record/${dir_name}/detection/scan_pretrain/detection_info.csv record/${dir_name}/detection/scan_pretrain/detection_info_${n_clean_sample}.csv || true
    fi
elif [[ "$def_name" == "sentinet" ]]; then
    if [ -f "record/${dir_name}/detection/sentinet_infer/detection_info_${n_clean_sample}.csv" ]; then
        echo "record/${dir_name}/detection/sentinet_infer/detection_info_${n_clean_sample}.csv" exists
    else
        python ./detection_infer/sentinet.py --result_file ${dir_name} --yaml_path ./config/detection/sentinet/${dataset}.yaml --dataset ${dataset} --clean_sample_num ${n_clean_sample} --target_layer ${senti_tlayer} --random_seed ${seed} ${bsize}
        mv record/${dir_name}/detection/sentinet_infer/detection_info.csv record/${dir_name}/detection/sentinet_infer/detection_info_${n_clean_sample}.csv || true
    fi
elif [[ "$def_name" == "vbd" ]]; then
    if [ -d "record/${dir_name}/defense/vbd" ]; then
        echo "record/${dir_name}/defense/vbd" exists
    else
        if [[ "$dataset" == "tiny" ]]; then
        extra_args=""
            #extra_args="--ensemble_detect_min_thr 1 --pois_model_arch 64P,10-10P,16-16P --pois_train_perc 0.8 --detector_train_perc 1.0"
        else
        extra_args=""
            #extra_args="--ensemble_detect_min_thr 1 --pois_model_arch 64P,10-10P,16-16P --pois_train_perc 0.8 --detector_train_perc 1.0"
        fi
        python ./defense/VarianceBasedDefense.py --result_file ${dir_name} --yaml_path ./config/defense/vbd/${dataset}.yaml --dataset ${dataset} --no_retrain True --random_seed ${seed} ${bsize} ${extra_args}
    fi
else
    echo "Unknown defense method ${def_name}"
    exit 1
fi


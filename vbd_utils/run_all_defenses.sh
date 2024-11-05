#!/usr/bin/env bash

# Software Name: BackdoorBench/run_all_defenses
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Run all the defenses planned for the evaluation

set -e

if [[ "${1}" == "dryrun" ]]; then
    CMD=echo
else
    CMD=""
fi

function run_defense () {
    def_name=$1
    dir_name=$2

    dataset=`cut -d'_' -f1 <<<"${dir_name}"`
    if [[ "$dataset" == "cifar10" ]]; then
        n_clean_sample=100
    elif [[ "$dataset" == "tiny" ]]; then
        n_clean_sample=2000
    else
        n_clean_sample=100
    fi

    case $def_name in
        "abl")
        output_name="detection_pretrain/abl/detection_info.csv" ;;
        "agpd")
        output_name="detection_pretrain/agpd/results_${n_clean_sample}.csv" ;;
        "scan")
        output_name="detection/scan_pretrain/detection_info_${n_clean_sample}.csv" ;;
        "sentinet")
        output_name="detection/sentinet_infer/detection_info_${n_clean_sample}.csv" ;;
        "vbd")
        output_name="defense/vbd" ;;
        *)
        echo "Unknown defense: ${def_name}" >&2 ; exit 1 ;;
    esac

    foname="record/${dir_name}/${output_name}"
    if [ -e "${foname}" ]; then
        echo "${foname} already exists (ignored)" >&2
    else
        if [[ "${CMD}" == "" ]]; then
            echo "Running vbd_utils/run_defense.sh $@ ..."
        fi
        ${CMD} vbd_utils/run_defense.sh $@
    fi
}

for l in `ls -d record/*_maskblended_* | xargs -n1 basename`
do
    if [ ! -f "record/${l}/attack_result.pt" ]; then
            echo "record/${l}/attack_result.pt does not exist (ignored)" >&2
    else
        for def_type in abl agpd scan vbd
        do
            run_defense $def_type ${l}
        done
    fi
done


#!/usr/bin/bash

# Software Name: BackdoorBench/sbatch_run_attack
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Run the attacks on a slurm env

#SBATCH --gpus-per-node=1
#SBATCH --output=vbd_utils/slurm/tmp/log/slurm-%A-%a-%j.out
#SBATCH --error=vbd_utils/slurm/tmp/log/slurm-%A-%a-%j.err


set -e

MAX_RUNNING_JOBS=10

if [[ "$SLURM_ARRAY_TASK_ID" != "" ]]; then
    if [[ "$1" == "" ]]; then
        echo "Job file not provided"
        exit 1
    fi
    job_file="$1"
    num=${SLURM_ARRAY_TASK_ID}
    cmd=$(sed "${num}q;d" ${job_file})
    echo "cmd #${num} from ${job_file}:"
    echo vbd_utils/slurm/run_srun.sh ${cmd}
    echo "..."
    vbd_utils/slurm/run_srun.sh ${cmd}
    exit 0
else
    SRUN_FNAME=vbd_utils/slurm/srun_args.local
    if [ ! -f "${SRUN_FNAME}" ]; then
        CONTAINER_IMAGE=pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
        CONTAINER_NAME=$(echo $CONTAINER_IMAGE | sed 's%[/:]%_%g')
        echo "Please set the srun arg variables in the file ${SRUN_FNAME}"
        echo "e.g.: --container-image=${CONTAINER_IMAGE} --container-name=${CONTAINER_NAME}"
        exit 1
    fi

    #double check to avoid recursion
    if [[ "$@" != "" ]]; then
        echo "No argument expected, received: $@" >&2
        exit 1
    fi

    job_file=vbd_utils/slurm/tmp/job/job-attack-$(date +%Y-%m-%d-%H-%M-%S).sh
    job_err=${job_file}.err
    vbd_utils/run_all_attacks.sh dryrun 1>${job_file} 2>${job_err}
    n_jobs=$(wc -l <${job_file})
    if [[ "${n_jobs}" > 0 ]]; then
        echo "Launching job array for ${job_file}"
        sbatch --array=1-${n_jobs}%${MAX_RUNNING_JOBS} vbd_utils/slurm/sbatch_run_attack.sh ${job_file}
    fi
fi


#!/bin/bash

# Software Name: BackdoorBench/run_srun
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: run a bash script on a slurm compute node

#SBATCH --time=24:00:00


#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=vbd_utils/slurm

SRUN_FNAME=${SCRIPT_DIR}/srun_args.local

if [[ -f "${SRUN_FNAME}" ]]; then
  CONTAINER_ARGS=$(<${SRUN_FNAME})
else
  CONTAINER_IMAGE=pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
  CONTAINER_NAME=$(echo $CONTAINER_IMAGE | sed 's%[/:]%_%g')
  echo "Please set the srun arg variables in the file ${SRUN_FNAME}"
  echo "e.g.: --container-image=${CONTAINER_IMAGE} --container-name=${CONTAINER_NAME}"
  exit 1
fi

CONTAINER_MOUNTS=./:/workdir

srun \
  --gpus-per-node=1 \
  ${CONTAINER_ARGS} \
  --container-mounts=$CONTAINER_MOUNTS \
  --container-workdir=/workdir \
  "$@"


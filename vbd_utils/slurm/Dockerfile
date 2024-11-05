# Software Name: BackdoorBench
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Docker file (to run on a slurm env)

FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Install some basic utilities.
RUN rm -f /etc/apt/sources.list.d/*.list \
&& apt-get update && apt-get install -y ca-certificates vim less emacs \
&& rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./setup.sh /install/
#RUN pip install -r /install/requirements.txt
RUN /install/setup.sh

#WORKDIR /app




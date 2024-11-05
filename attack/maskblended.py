#!/usr/bin/env python3

# Software Name: BackdoorBench/MaskBlended
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Blended Attack implementation with a binary mask


import argparse
import os
import sys

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args


class MaskBlended(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--attack_trigger_img_path", type=str, )
        parser.add_argument("--attack_trigger_mask_path", type=str, default='')
        parser.add_argument("--attack_train_blended_alpha", type=float, )
        parser.add_argument("--attack_test_blended_alpha", type=float, )
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/maskblended/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser


if __name__ == '__main__':
    attack = MaskBlended()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()

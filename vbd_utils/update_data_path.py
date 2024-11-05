#!/usr/bin/env python3

# Software Name: BackdoorBench/update_data_path
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Update the image data absolute path with a relative path in the saved pickle file

import os, sys, torch

if './' not in sys.path:
    sys.path.append('./')

from utils.save_load_attack import load_attack_result, save_attack_result


def update_img_path(result_fname):
    rpath=f'record/{result_fname}/attack_result.pt'
    data_dict = torch.load(rpath)
    changed = False
    d = [data_dict]
    while len(d)>0:
        t = d.pop()
        for k,v in t.items():
            if type(v)==dict:
                d.append(v)
            elif (type(v)==str) and (result_fname in v):
                v2 = './'+ v[v.index('record/'):]
                if v2 != v:
                    t[k] = v2
                    changed = True
    if changed:
        os.rename(rpath, rpath+'.old')
        torch.save(data_dict, rpath)
        print(f'{rpath} updated')

mp = 'record/'
for d in sorted(os.listdir(mp)):
    fname= mp + d +'/attack_result.pt'
    if os.path.isfile(fname):
        update_img_path(d)

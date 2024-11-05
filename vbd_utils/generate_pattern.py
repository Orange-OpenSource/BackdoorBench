#!/usr/bin/env python3

# Software Name : BackdoorBench/generate_pattern
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Generate patterns at different locations according to the target class

import argparse, math
import os, sys
import numpy as np
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--width', type=int, required=True, help='image width')
    parser.add_argument('--height',type=int, required=True, help='image height')
    parser.add_argument('--size', type=int, required=True, help='pattern size')
    parser.add_argument('--type', default='square',  choices=['square','grid','multi_color_grid'], help='type of pattern to generate')
    parser.add_argument('--color', default='255,255,255', type=str, help='color format: 255,0,0')
    parser.add_argument('--num_labels', type=int, required=True, help='total number of labels')
    parser.add_argument('--start_label', type=int, required=True, help='the label index from which to start')
    parser.add_argument('--group_labels', type=int, required=True, help='number of labels to group together')
    parser.add_argument('--shift_label', type=int, required=True, help='label i will be poisoned to (i+shift_label) mod number of labels')
    args = parser.parse_args()

    num_img = (args.num_labels - args.start_label)/args.group_labels
    if int(num_img)!=num_img:
        print(f'Error: number of patterns is not an int {num_img}. Adjust the parameters --num_labels, --start_label and --group_labels')
        sys.exit(-1)
    num_img = int(num_img)

    gw = math.floor(args.width/args.size)
    gh = math.floor(args.height/args.size)

    if num_img>(gw*gh):
        print(f'Error num_img is too large for width*height: {num_img}>{gw*gh}. Please adjust the parameters --num_labels, --start_label and --group_labels')
        sys.exit(-1)

    color = [min(int(i),255) for i in args.color.split(',')]

    out_dir = args.output + '/pat/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for i in range(num_img):
        #start from the bottom right corner
        x  = args.width - (math.floor(i%gw)*args.size) - args.size
        y  = args.height - (math.floor(i/gh)*args.size) - args.size
        img = np.zeros((args.height,args.width,3))
        img_mask = np.zeros((args.height,args.width,3))
        if args.type == 'grid':
            step = math.floor(args.size/3)
            img[y:y+step, x:x+step,:] = color
            img[y:y+step,x+2*step:x+3*step,:] = color
            img[y+step:y+2*step,x+step:x+2*step,:] = color
            img[y+2*step:y+3*step,x:x+step,:] = color
            img[y+2*step:y+3*step,x+2*step:x+3*step,:] = color
            img_mask[y:y+step,x:x+step,:] = 255
            img_mask[y:y+step,x+2*step:x+3*step,:] = 255
            img_mask[y+step:y+2*step,x+step:x+2*step,:] = 255
            img_mask[y+2*step:y+3*step,x:x+step,:] = 255
            img_mask[y+2*step:y+3*step,x+2*step:x+3*step,:] = 255
        elif args.type == 'multi_color_grid':
            step = math.floor(args.size/3)
            img[y:y+step,x:x+step,:] = [0,255,0]
            img[y:y+step,x+2*step:x+3*step,:] = [255,0,0]
            img[y+step:y+2*step,x+step:x+2*step,:] = [255,255,0]
            img[y+2*step:y+3*step,x:x+step,:] = [255,0,255]
            img[y+2*step:y+3*step,x+2*step:x+3*step,:] = [0,0,255]
            img_mask[y:y+step,x:x+step,:] = 255
            img_mask[y:y+step,x+2*step:x+3*step,:] = 255
            img_mask[y+step:y+2*step,x+step:x+2*step,:] = 255
            img_mask[y+2*step:y+3*step,x:x+step,:] = 255
            img_mask[y+2*step:y+3*step,x+2*step:x+3*step,:] = 255
        else:
            step = math.floor(args.size/3)
            img[y:y+3*step,x:x+3*step,:] = color
            img_mask[y:y+3*step,x:x+3*step,:] = 255
        Image.fromarray(img.astype(np.uint8), 'RGB').save(f'{out_dir}/{i}.png')
        Image.fromarray(img_mask.astype(np.uint8), 'RGB').save(f'{out_dir}/{i}_mask.png')
    
    #now generate the links
    group_max = num_img #number of group
    all_target_labels = [i for i in range(args.start_label)] + [args.start_label+(i*args.group_labels) for i in range(group_max)]
    for group_i in range(group_max):
        label_group_start = args.start_label +  group_i*args.group_labels
        #target_label = (label_group_start + args.shift_label) % args.num_labels
        target_label = all_target_labels[ (args.start_label+group_i+ args.shift_label) % len(all_target_labels) ]
        for j in range(args.group_labels):
            label_i = label_group_start + j
            os.symlink(f'pat/{group_i}.png', f'{args.output}/{label_i}_{target_label}.png')
            os.symlink(f'pat/{group_i}_mask.png', f'{args.output}/{label_i}_{target_label}_mask.png')






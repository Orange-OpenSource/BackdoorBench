#!/usr/bin/env python3

# Software Name: BackdoorBench/perf_reader
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This software is distributed under the Creative Commons Attribution Non Commercial 4.0 International,
# see the "LICENSE.txt" file for more details
#
# Authors: Sujeevan Aseervatham
# Software description: Aggregate the perf. results from the defense methods

import os
from datetime import datetime
import numpy as np
import pandas as pd

def read_df_summary(filename):
    try:
        with open(filename, 'r') as f:
            t = f.readlines()
            assert(t[0].strip().split(',')[-3:] == ['test_acc','test_asr','test_ra'])
            r = t[1].strip().split(',')
            assert(r[0]=='last')
            r = (str(round(float(i), 4)) for i in r[-3:])
    except Exception as e:
        print(f'Error with {filename} : {e}')
        r = ('', '', '')
    return r

def read_vbd_detect_perf(path):
    fn = path + '/train_perf_poison_detection.csv'
    try:
        with open(fn, 'r') as f:
            t = [i.strip() for i in f.readlines()]
            npats =  len(t)-2
            s = t[-1]
            i = s.index(',')
            assert(s[:i]=='ALL')
            d = eval("{'"+s[i+1:].replace(':', "':").replace(',', ",'")+"}")
            tp, tn, fp, fn = d['tp'], d['tn'], d['fp'], d['fn']
            tpr = tp / (tp+fn) if (tp+fn) != 0 else 0 
            fpr =  fp / (fp + tn)  if (fp+tn) != 0 else 0 
            precision = tp/(tp+fp) if (tp+fp) != 0 else 0 
            recall = tp/(tp+fn) if (tp+fn) != 0 else 0 
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
            acc = (tp+tn)/(tp+fn+tn+fp) if (tp+fn+tn+fp) != 0 else 0 
            #N_Patterns,ACC,F1,Precision,Recall,TPR,FPR,TP,FP,TN,FN
            r = [npats]+ [acc, f1, precision, recall] + [tpr, fpr, tp, fp, tn, fn]
            r = [str(round(i,4)) for i in r]
    except Exception as e:
        print(f'Error with {fn} : {e}')
        r = ['']*11
    return r

def read_agpd_detect_perf(filename):
    fn = filename
    try:
        with open(fn, 'r') as f:
            t = [i.strip() for i in f.readlines()]
            npats =  ''
            s = t[0].split(',')
            if s[10] == 'clean model':
                s[10] = '0.'
            tn, fp, fn, tp, tpr, fpr, f1, auc = int(float(s[4])), int(float(s[5])), int(float(s[6])), int(float(s[7])), float(s[8]), float(s[9]), float(s[10]), float(s[11])
            precision = tp/(tp+fp) if (tp+fp) != 0 else 0 
            recall = tp/(tp+fn) if (tp+fn) != 0 else 0 
            acc = (tp+tn)/(tp+fn+tn+fp) if (tp+fn+tn+fp) != 0 else 0 
            #N_Patterns,ACC,F1,Precision,Recall,TPR,FPR,TP,FP,TN,FN
            r = [npats] + [str(round(i,4)) for i in [acc, f1, precision, recall, tpr, fpr, tp, fp, tn, fn]]
    except Exception as e:
        print(f'Error with {fn} : {e}')
        r = ['']*11
    return r

def read_abl_detect_perf(filename):
    fn = filename
    try:
        with open(fn, 'r') as f:
            t = [i.strip() for i in f.readlines()]
            npats =  ''
            s = t[1].split(',')
            tn, fp, fn, tp, tpr, fpr, auc = int(float(s[1])), int(float(s[2])), int(float(s[3])), int(float(s[4])), float(s[5]), float(s[6]), float(s[7])
            precision = tp/(tp+fp) if (tp+fp) != 0 else 0 
            recall = tp/(tp+fn) if (tp+fn) != 0 else 0
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
            acc = (tp+tn)/(tp+fn+tn+fp) if (tp+fn+tn+fp) != 0 else 0 
            #N_Patterns,ACC,F1,Precision,Recall,TPR,FPR,TP,FP,TN,FN
            r = [npats] + [str(round(i,4)) for i in [acc, f1, precision, recall, tpr, fpr, tp, fp, tn, fn]]
    except Exception as e:
        print(f'Error with {fn} : {e}')
        r = ['']*11
    return r

def read_std_detect_perf(filename):
    fn = filename
    try:
        with open(fn, 'r') as f:
            t = [i.strip() for i in f.readlines()]
            npats =  ''
            s = t[1].split(',')
            tn, fp, fn, tp, tpr, fpr = int(float(s[1])), int(float(s[2])), int(float(s[3])), int(float(s[4])), float(s[5]), float(s[6])
            precision = tp/(tp+fp) if (tp+fp) != 0 else 0 
            recall = tp/(tp+fn) if (tp+fn) != 0 else 0 
            f1 = (2.*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
            acc = (tp+tn)/(tp+fn+tn+fp)  if (tp+fn+tn+fp) != 0 else 0 
            #N_Patterns,ACC,F1,Precision,Recall,TPR,FPR,TP,FP,TN,FN
            r = [npats] + [str(round(i,4)) for i in [acc, f1, precision, recall, tpr, fpr, tp, fp, tn, fn]]
    except Exception as e:
        print(f'Error with {fn} : {e}')
        r = ['']*11
    return r


def read_model_perf(filename):
    fn = filename
    try:
        with open(fn, 'r') as f:
            t = [i.strip() for i in f.readlines()]
            s = t[1].split(',')
            train_acc, train_acc_clean_only, train_asr_bd_only =float(s[4]), float(s[5]), float(s[6])
            test_acc, test_asr = float(s[10]), float(s[11])
            r = [str(round(i,4)) for i in [train_acc, train_acc_clean_only, train_asr_bd_only, test_acc, test_asr] ]
    except Exception as e:
        print(f'Error with {fn} : {e}')
        r = ['']*5
    return r

mp = 'record/'
lines = []
perf_lines = []
model_perf_lines = []
for d in sorted(os.listdir(mp)):
    t = d.split('_')
    if len(t)>2 and t[1] == 'maskblended':
        dataset, attack, cl, patt, fact, seed = t[0], t[1], '_'.join(t[2:4]), '_'.join(t[4:-2]), t[-2], t[-1]
        #dataset, patt, cl, fact, defense, test_acc, test_asr, test_racc
        defense = 'no_defense'
        test_acc, test_asr, test_racc = read_df_summary(mp+d+'/attack_df_summary.csv')
        lines.append(','.join([dataset, patt, cl, fact, seed, defense, test_acc, test_asr, test_racc]))
        defense = 'vbd_relabel'
        test_acc, test_asr, test_racc = read_df_summary(mp+d+'/defense/vbd/vbd_df_summary.csv')
        lines.append(','.join([dataset, patt, cl, fact, seed, defense, test_acc, test_asr, test_racc]))
        defense = 'vbd_suppress'
        test_acc, test_asr, test_racc = read_df_summary(mp+d+'/defense/vbd/res_suppress/vbd_df_summary.csv')
        lines.append(','.join([dataset, patt, cl, fact, seed, defense, test_acc, test_asr, test_racc]))
        defense = 'abl'
        test_acc, test_asr, test_racc = read_df_summary(mp+d+'/defense/abl/abl_df_summary.csv')
        lines.append(','.join([dataset, patt, cl, fact, seed, defense, test_acc, test_asr, test_racc]))
        a = read_vbd_detect_perf(mp+d+'/defense/vbd')
        perf_lines.append(','.join([dataset, patt, cl, fact, seed, 'vbd']+a))
        for n_clean_sample in [100, 2000]:
            a = read_agpd_detect_perf(mp+d+f'/detection_pretrain/agpd/results_{n_clean_sample}.csv')
            perf_lines.append(','.join([dataset, patt, cl, fact, seed, f'agpd_{n_clean_sample}']+a))
            a = read_std_detect_perf(mp+d+f'/detection/scan_pretrain/detection_info_{n_clean_sample}.csv')
            perf_lines.append(','.join([dataset, patt, cl, fact, seed, f'scan_{n_clean_sample}']+a))
        a = read_abl_detect_perf(mp+d+f'/detection_pretrain/abl/detection_info.csv')
        perf_lines.append(','.join([dataset, patt, cl, fact, seed, 'abl']+a))
        
        a = read_model_perf(mp+d+'/attack_df_summary.csv')
        model_perf_lines.append(','.join([dataset, patt, cl, fact, 'original poisoned_model']+a))
        
    
with open('perf_summary.csv', 'w') as f:
    f.write('Dataset,Pattern,Target_Class,Blending,seed,Defense,ACC,ASR,Robust_ACC'+'\n')
    f.write('\n'.join(lines)+'\n')
with open('detect_summary.csv', 'w') as f:
    f.write('Dataset,Pattern,Target_Class,Blending,seed,Defense,N_Patterns,ACC,F1,Precision,Recall,TPR,FPR,TP,FP,TN,FN'+'\n')
    f.write('\n'.join(perf_lines)+'\n')

with open('model_summary.csv', 'w') as f:
    f.write('Dataset,Pattern,Target_Class,Blending,seed,Model,train_acc,train_acc_clean,train_asr_bd,test_acc,test_asr'+'\n')
    f.write('\n'.join(model_perf_lines)+'\n')


#create a summary file

df = pd.read_csv('detect_summary.csv')
if 'seed' not in df.columns:
    df['seed']=0
    df = df[list(df.columns[:4])+ ['seed'] + list(df.columns[4:-1])]

metrics = ['F1', 'Precision', 'Recall']
d = df.pivot_table(metrics, ['Dataset', 'Pattern','Target_Class','Blending', 'seed'], 'Defense')
d.columns = d.columns.map('_'.join)
d = d.reset_index()
for i in ['agpd','scan']:
    for j in metrics:
        d[j+'_'+i] = 0
        if(j+'_'+i+'_100') in d.columns:
            d[j+'_'+i] += d[j+'_'+i+'_100'].fillna(0)
        if(j+'_'+i+'_2000') in d.columns:
            d[j+'_'+i] += d[j+'_'+i+'_2000'].fillna(0)
m = [j+'_'+i for i in ['vbd', 'abl', 'agpd', 'scan'] for j in metrics]
d = d[['Dataset', 'Pattern', 'Target_Class', 'Blending', 'seed']+m].fillna(0)
df = d


def summ_info(df, target_class_list, blend_list, aggcol, dropcol, desc, global_avg=True, sum_only=False, t_display=False):
    d = df[df['Target_Class'].isin(target_class_list) & df['Blending'].isin(blend_list)]
    buffer = []
    if sum_only==False:
        if global_avg:
            d2 = d.drop(columns=['Target_Class', 'seed', dropcol])
        else:
            d2 = d.drop(columns=['Target_Class', dropcol]).groupby(['Dataset', 'seed', aggcol], as_index=False).mean()
            d2 = d2.drop(columns=['seed'])
        d2 = d2.groupby(['Dataset', aggcol]).agg(['mean', 'std'])
        d2.columns = d2.columns.map('_'.join)
        d2 = d2.reset_index()
        d2.sort_values(by=['Dataset', aggcol])
        if aggcol=='Pattern':
            d2['Pattern'] = d2['Pattern'].apply(lambda x: x.replace('trigger_', ''))
        buffer += ['', '', desc]
        buffer += [','.join(d2.columns)] + [','.join([str(k) for k in d2.iloc[l].values]) for l in range(len(d2))]
        #if t_display:
        #    print(desc)
        #    display(d2)
    else:
        buffer += ['', '', desc]
        #if t_display:
        #    print(desc)
    if global_avg:
        d2 = d.drop(columns=['Target_Class', 'Pattern', 'seed', 'Blending'])
    else:
        d2 = d.drop(columns=['Target_Class', 'Pattern', 'Blending']).groupby(['Dataset', 'seed'], as_index=False).mean()
        d2 = d2.drop(columns=['seed'])
    d2 = d2.groupby(['Dataset']).agg(['mean', 'std'])
    d2.columns = d2.columns.map('_'.join)
    d2 = d2.reset_index()
    d2.sort_values(by=['Dataset'])
    d2[aggcol] = 'average'
    davg = d2[['Dataset', aggcol]+list(d2.columns[1:-1])]

    if global_avg:
        d2 = d.drop(columns=['Target_Class', 'Pattern', 'seed', 'Dataset', 'Blending'])
    else:
        d2 = d.drop(columns=['Target_Class', 'Pattern', 'Dataset', 'Blending']).groupby(['seed'], as_index=False).mean()
        d2 = d2.drop(columns=['seed'])
    d2['k']=0
    d2 = d2.groupby(['k'], as_index=False).agg(['mean', 'std'])
    d2.columns = d2.columns.map('_'.join)
    d2 = d2.reset_index(drop=True).drop(columns=['k_'])
    d2['Dataset'] = 'average'
    d2[aggcol] = ''
    d2 = d2[['Dataset', aggcol]+list(d2.columns[0:-2])]
    d2 = pd.concat([davg, d2]).reset_index(drop=True)
    if sum_only==True:
        buffer += [','.join(d2.columns)] 
    buffer += [','.join([str(k) for k in d2.iloc[l].values]) for l in range(len(d2))]
    #if t_display:
    #    display(d2)
    return buffer


buffer = []
t_display=False
global_avg=False
t = [i for i in df['Target_Class'].unique() if i.endswith('_all')]
t2 = [i for i in df['Blending'].unique() if i==1]
buffer +=summ_info(df, target_class_list=t, blend_list=t2, aggcol='Pattern', dropcol='Blending', desc='all-to-one - BadNets', global_avg=global_avg, t_display=t_display)

t = ['1_1']
t2 = [i for i in df['Blending'].unique() if i==1]
buffer +=summ_info(df, target_class_list=t, blend_list=t2, aggcol='Pattern', dropcol='Blending', desc='all-to-all - BadNets', global_avg=global_avg, t_display=t_display)


t = [i for i in df['Target_Class'].unique() if i.endswith('_all')]
t2 = [i for i in df['Blending'].unique() if i!=1]
buffer +=summ_info(df, target_class_list=t, blend_list=t2, aggcol='Blending', dropcol='Pattern', desc='all-to-one - Blended', global_avg=global_avg, t_display=t_display)

t = ['1_1']
t2 = [i for i in df['Blending'].unique() if i!=1]
buffer +=summ_info(df, target_class_list=t, blend_list=t2, aggcol='Blending', dropcol='Pattern', desc='all-to-all - Blended', global_avg=global_avg, t_display=t_display)


t = [i for i in df['Target_Class'].unique() if i.endswith('_all')]+['1_1']
t2 = [i for i in df['Blending'].unique()]
buffer +=summ_info(df, target_class_list=t, blend_list=t2, aggcol='Blending', dropcol='Pattern', desc='Global avg', sum_only=True, global_avg=global_avg, t_display=t_display)

with open("detect_summary_avg.csv", 'w') as f:
    f.write('\n'.join(buffer))


print('All done')


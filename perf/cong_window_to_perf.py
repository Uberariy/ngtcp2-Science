"""
This program is an .ipybn notebook utility, used to convert special data.

"""

# %%
from __future__ import annotations
from json.tool import main
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import json
import yaml
import os
from collections import defaultdict
sns.set_theme(style="whitegrid")

# %%
'''Functions, that process files with multiple statlog.py runs outputs into data.'''
def convert_speed(fl):
    '''Converts big float speeds into 2^n format string'''
    if fl >= 1024**3//8:
        return f"{int(fl*8//(1024**3))} Gbit/s"
    if fl >= 1024**2//8:
        return f"{int(fl*8//(1024**2))} Mbit/s"
    if fl >= 1024//8:
        return f"{int(fl*8//(1024))} Kbit/s"

def convert_speed_to_mbit(fl):
    '''Converts big float speeds into Mbit format'''
    return int(fl*8//(1024**2))

def convert_speed_to_kbit(fl):
    '''Converts big float speeds into Kbit format'''
    return int(fl*8//1024)

def calculate_accuracy(th, pr):
    '''Calculates accuracy'''
    return math.fabs((th - pr) / th)

def get_data_yaml(paths, data_format_type):
    '''
    Each file in each dir is:

{additional_info: {channel_bw: 180, channel_congestion_control: bbrfrcst, channel_jitt: 1,
    channel_loss: 0.001, channel_rtt: 20, cong_window: 5000000}, content: [{'bytes sent: ': 23552551,
      'mean cwnd: ': 5000000.0, 'mean jitter: ': 0.238482, 'mean loss2: ': 0.951885,
      'mean loss: ': 31.848659, 'mean s_rtt: ': 51.163866, time: 0.0-1.0}, {'bytes sent: ': 23552551,
      'mean cwnd: ': 5000000.0, 'mean jitter: ': 0.238482, 'mean loss2: ': 0.951885,
      'mean loss: ': 31.848659, 'mean s_rtt: ': 51.163866, time: global}]}
    '''
    data = []
    good_data_amount = 0
    bad_data_amount = 0
    file_idx = 0
    for directory in paths:
        for filename in os.listdir(directory):
            file_idx += 1
            dir_file = os.path.join(directory, filename)
            if os.path.isfile(dir_file) and (filename != 'TooLongERRORs'):
                with open(dir_file, 'r') as f:
                    '''Analyse a single experiment'''
                    data_sample = []
                    datadict = yaml.safe_load(f)
                    '''Add parameters'''
                    conj_c = datadict['additional_info']['channel_congestion_control'].upper()
                    data_sample.append(conj_c)
                    theor_rtt = datadict['additional_info']['channel_rtt']
                    data_sample.append(theor_rtt)
                    theor_loss_percent = datadict['additional_info']['channel_loss']
                    data_sample.append(theor_loss_percent)
                    theor_speed_in_kbit = datadict['additional_info']['channel_bw'] * 1024
                    data_sample.append(theor_speed_in_kbit)
                    theor_jitt = datadict['additional_info']['channel_jitt']
                    data_sample.append(theor_jitt)
                    theor_cong_wind = int(datadict['additional_info']['cong_window'])
                    data_sample.append(theor_cong_wind)
                    parti = float(datadict['content'][0]['time'].split('-')[1]) - float(datadict['content'][0]['time'].split('-')[0])
                    '''The statlog.py puts global data in the end of the list => [-1]'''
                    real_speed_in_kbit = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
                    data_sample.append(real_speed_in_kbit)
                    real_rtt = datadict['content'][-1]['mean s_rtt: ']
                    data_sample.append(real_rtt)
                    real_loss1 = datadict['content'][-1]['mean loss: ']
                    data_sample.append(real_loss1)
                    real_loss2 = datadict['content'][-1]['mean loss2: ']
                    data_sample.append(real_loss2)
                    real_jitt = datadict['content'][-1]['mean jitter: ']
                    data_sample.append(real_jitt)
                    real_cong_wind = int(datadict['content'][-1]['mean cwnd: '])
                    data_sample.append(real_cong_wind)
                    content_length = len(datadict['content']) - 1 # Because 1 goes to calculated means
                    experiment_duration = float(content_length) * parti # In seconds
                    data_sample.append(experiment_duration)
                '''Skip if bad data (experiment was too fast). 
                Minimum experiment duration is currently 1min, accuracy is 1 second.'''
                if (experiment_duration < 0) or (real_loss1 > 2000):
                    print(filename)
                    bad_data_amount += 1
                    continue
                else:
                    good_data_amount += 1
                
                data.append(data_sample)
                if (file_idx % 100 == 0):
                    print(f"File number {file_idx} / ...")

    col = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)",
           "Channel Jitter (ms)", "Preset Congestion Window (bytes)", "Sender Speed (Kbit/s)", 
           "Sender RTT (ms)", "Sender lost data to data inflight (%)", "Sender lost data to data sent (%)", 
           "Sender Jitter (ms)", "Sender Congestion Window (bytes)", "Experiment duration (sec)"]

    print("Good data samples amount: ", good_data_amount, " bad: ", bad_data_amount)
    good_data_amount = 1 if good_data_amount == 0 else good_data_amount
    return data, col, bad_data_amount / good_data_amount

# %%
'''Define pathes'''

'''Path to save csv'''
path_save_csv = 'csv_stat'

'''data_format_type equals following numbers depending on the task:
1. Speed calculation on congestion window;
'''
data_format_type = 1
paths_yml_dir = ['perfres_w3_speed_v3']

# %%
'''Put files needed to be composed into a dataframe in statfiles, then specify algos, that we want to analyse.'''
if __name__ == "__main__":
    data_lists, col, bad_data_fraction1 = get_data_yaml(paths_yml_dir, data_format_type=data_format_type)
    df = pd.DataFrame(data_lists, columns=col)

# %%
'''If we want to group by something and calculate means'''
if __name__ == "__main__":
    pass

# %%
def explore_cong_window_in_one_dot(df_dot, dot, channel_features):
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))
    fig.suptitle(f'Dependencies between BBR Congestion Window and features.\nRTT = {dot[1]} ms, Loss = {dot[2]} %, BW = {dot[3]} Kbit/s.')
    sns.lineplot(ax=axes[0],data=df_dot, x="Preset Congestion Window (bytes)", y=f"Sender Speed (Kbit/s)", ci=70)
    sns.lineplot(ax=axes[1],data=df_dot, x="Preset Congestion Window (bytes)", y=f"Sender lost data to data sent (%)", ci=70)
    sns.lineplot(ax=axes[2],data=df_dot, x="Preset Congestion Window (bytes)", y=f"Sender RTT (ms)", ci=70)
    # sns.lineplot(ax=axes[3],data=df_dot, x="Preset Congestion Window (bytes)", y=f"Experiment duration (sec)", ci=70)

if __name__ == "__main__":
    channel_features = ["Congestion Controller", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)", "Channel Jitter (ms)"]
    dots = list(df[channel_features].value_counts().index)
    tmp = 0
    for dot in dots:
        df_dot = df
        tmp += 1
        for feature in channel_features:
            df_dot = df_dot.loc[(df_dot[feature] == dot[channel_features.index(feature)])]
        if tmp >= 0:
            print("Dot:", df_dot, "Shape:", df_dot.shape)
            explore_cong_window_in_one_dot(df_dot, dot, channel_features)

# %%
df.loc[df["Sender lost data to data sent (%)"] > 100]
# %%
df
# %%

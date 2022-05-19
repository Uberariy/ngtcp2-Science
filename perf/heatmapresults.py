"""
This program is an .ipybn notebook utility.

"""

#%%
from json.tool import main
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

#%%
def convert_speed(fl):
    '''Converts big float speeds into 2^n format string'''
    if fl >= 1024**3//8:
        return f"{int(fl*8//(1024**3))} Gbit/s"
    if fl >= 1024**2//8:
        return f"{int(fl*8//(1024**2))} Mbit/s"
    if fl >= 1024//8:
        return f"{int(fl*8//(1024))} Kbit/s"

def get_data(paths):
    sla_d = defaultdict(float)
    anno_d = defaultdict(dict)
    maintext = ""
    for path in paths:
        with open(path, 'r') as f:
            text = f.read()
        maintext += text
    patt = re.compile(r"BBRFRCSTexperiment. (.*)(\n|.)*?Mean speed (.*)\n?If it is bbrfrcst: (.*)")
    for i in patt.findall(maintext):
        p_rtt = float(i[0].split()[1])
        p_loss = float(i[0].split()[3])
        p_bw = float(i[0].split()[5])
        real_bw = float(i[2].split()[0]) / float(i[2].split()[3])
        if "samples" in anno_d[(p_rtt, p_bw)]:
            '''Number of experiments'''
            anno_d[(p_rtt, p_bw)]["samples"] += 1
        else:
            anno_d[(p_rtt, p_bw)]["samples"] = 1
        if "loss" in anno_d[(p_rtt, p_bw)]:
            '''It is mean loss set in channel (by experiments)'''
            anno_d[(p_rtt, p_bw)]["p_loss"] = anno_d[(p_rtt, p_bw)]["p_loss"] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
            anno_d[(p_rtt, p_bw)]["p_loss"] += p_loss / anno_d[(p_rtt, p_bw)]["samples"]
        else:
            anno_d[(p_rtt, p_bw)]["p_loss"] = p_loss
        if "real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is mean real bw (by experiments)'''
            anno_d[(p_rtt, p_bw)]["real_bw"] = anno_d[(p_rtt, p_bw)]["real_bw"] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
            anno_d[(p_rtt, p_bw)]["real_bw"] += real_bw / anno_d[(p_rtt, p_bw)]["samples"]
        else:
            anno_d[(p_rtt, p_bw)]["real_bw"] = real_bw
        if "min_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is min real bw (by experiments)'''
            if real_bw < anno_d[(p_rtt, p_bw)]["min_real_bw"]:
                anno_d[(p_rtt, p_bw)]["min_real_bw"] = real_bw
        else:
            anno_d[(p_rtt, p_bw)]["min_real_bw"] = real_bw
        if "max_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is max real bw (by experiments)'''
            if real_bw > anno_d[(p_rtt, p_bw)]["max_real_bw"]:
                anno_d[(p_rtt, p_bw)]["max_real_bw"] = real_bw
        else:
            anno_d[(p_rtt, p_bw)]["max_real_bw"] = real_bw
        # print(anno_d[(p_rtt, p_bw)]["samples"])
        sla_d[(p_rtt, p_bw)] = sla_d[(p_rtt, p_bw)] * (anno_d[(p_rtt, p_bw)]["samples"] - 1) / anno_d[(p_rtt, p_bw)]["samples"]
        if i[3] == "SLA IS OKAY":
            sla_d[(p_rtt, p_bw)] += 1 / anno_d[(p_rtt, p_bw)]["samples"]
    patt = re.compile(r"BBR2experiment. (.*)(\n|.)*?Mean speed (.*)\n")
    for i in patt.findall(maintext):
        # print(i)
        p_rtt = float(i[0].split()[1])
        p_loss = float(i[0].split()[3])
        p_bw = float(i[0].split()[5])
        real_bbr_bw = float(i[2].split()[0]) / float(i[2].split()[3])
        '''Next condition can be ignored'''
        if (p_rtt, p_bw) not in anno_d:
            continue
        if "bbr_samples" in anno_d[(p_rtt, p_bw)]:
            '''Number of experiments'''
            anno_d[(p_rtt, p_bw)]["bbr_samples"] += 1
        else:
            anno_d[(p_rtt, p_bw)]["bbr_samples"] = 0
        if "bbr_loss" in anno_d[(p_rtt, p_bw)]:
            '''It is mean loss set in channel (by experiments)'''
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] = anno_d[(p_rtt, p_bw)]["bbr_p_loss"] * (anno_d[(p_rtt, p_bw)]["bbr_samples"] - 1) / anno_d[(p_rtt, p_bw)]["bbr_samples"]
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] += p_loss / anno_d[(p_rtt, p_bw)]["bbr_samples"]
        else:
            anno_d[(p_rtt, p_bw)]["bbr_p_loss"] = p_loss
        if "bbr_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is mean real bw (by experiments)'''
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] = anno_d[(p_rtt, p_bw)]["bbr_real_bw"] * (anno_d[(p_rtt, p_bw)]["bbr_samples"] - 1) / anno_d[(p_rtt, p_bw)]["bbr_samples"]
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] += real_bbr_bw / anno_d[(p_rtt, p_bw)]["bbr_samples"]
        else:
            anno_d[(p_rtt, p_bw)]["bbr_real_bw"] = real_bbr_bw
        if "bbr_min_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is min real bw (by experiments)'''
            if real_bbr_bw < anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]:
                anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"] = real_bbr_bw
        else:
            anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"] = real_bbr_bw
        if "bbr_max_real_bw" in anno_d[(p_rtt, p_bw)]:
            '''It is max real bw (by experiments)'''
            if real_bbr_bw > anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]:
                anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"] = real_bbr_bw
        else:
            anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"] = real_bbr_bw
        anno_d[(p_rtt, p_bw)]["annotation1"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\nBBR2 speed:         {}".format(
            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
            anno_d[(p_rtt, p_bw)]["samples"],
            anno_d[(p_rtt, p_bw)]["p_loss"],
            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
        )
        anno_d[(p_rtt, p_bw)]["annotation2"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\n(min: {}, max: {})\nBBR2 speed:         {}\n(min: {}, max: {})".format(
            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
            anno_d[(p_rtt, p_bw)]["samples"],
            anno_d[(p_rtt, p_bw)]["p_loss"],
            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["max_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]),
        )
        anno_d[(p_rtt, p_bw)]["annotation3"] = "SLA: {}/{}\nChannel loss: {}\nBBRFRCST speed: {}\n(min: {}, max: {})\nBBR2 speed:         {}\n(min: {}, max: {})".format(
            int(sla_d[(p_rtt, p_bw)] * anno_d[(p_rtt, p_bw)]["samples"]),
            anno_d[(p_rtt, p_bw)]["samples"],
            anno_d[(p_rtt, p_bw)]["p_loss"],
            convert_speed(anno_d[(p_rtt, p_bw)]["real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["max_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_min_real_bw"]),
            convert_speed(anno_d[(p_rtt, p_bw)]["bbr_max_real_bw"]),
        )
        # print(anno_d[(p_rtt, p_bw)]["annotation"])
    return sla_d, anno_d


#%%
# INPUT:
statfiles = ["perfres_i1_1dot15", "perfres_bbr2_lost0.1"] # Name of file
# statfiles = ["perfres_i1_1dot20", "perfres_bbr2_lost0.1"] # Name of file

sla, anno = get_data(statfiles)

# %%
rtts = set()
bws = set()
for i in sla.keys():
    rtts.add(i[0])
    bws.add(i[1])
l_rtts = sorted(list(rtts))
l_bws = sorted(list(bws))
pd_sla = []
pd_anno = []
for i in l_rtts:
    tmp_l = []
    tmp_a_l = []
    for j in l_bws:
        tmp_l.append(sla[(i, j)])
        if "annotation1" in anno[(i, j)]:
            tmp_l[-1] += 1
            tmp_a_l.append(anno[(i, j)]["annotation1"])
        else:
            tmp_a_l.append("No data")
    pd_anno.append(tmp_a_l)
    pd_sla.append(tmp_l)
pd2_sla = pd.DataFrame(pd_sla, columns = l_bws, index = l_rtts)
pd2_anno = pd.DataFrame(pd_anno)
# print(pd_anno, pd_sla, np.array(pd_sla).size, np.array(pd_anno).size)
pd_anno = pd.Series(anno).reset_index()
plt.figure(figsize=(17, 13))
sns.heatmap(pd2_sla, annot=pd2_anno, fmt="", center=0, linewidths=.5)
plt.title('SLA Goodness Statistics')
plt.xlabel('Channel BW')
plt.ylabel('Channel RTT')

# %%
rtts = set()
bws = set()
for i in sla.keys():
    rtts.add(i[0])
    bws.add(i[1])
l_rtts = sorted(list(rtts))
l_bws = sorted(list(bws))
pd_sla = []
pd_anno = []
for i in l_rtts:
    tmp_l = []
    tmp_a_l = []
    for j in l_bws:
        if "annotation1" in anno[(i, j)]:
            tmp_l.append(anno[(i, j)]["real_bw"] / anno[(i, j)]["bbr_real_bw"])
            tmp_a_l.append(anno[(i, j)]["annotation1"])
        else:
            tmp_l.append(0)
            tmp_a_l.append("No data")
    pd_anno.append(tmp_a_l)
    pd_sla.append(tmp_l)
pd2_sla = pd.DataFrame(pd_sla, columns = l_bws, index = l_rtts)
pd2_anno = pd.DataFrame(pd_anno)
# print(pd_anno, pd_sla, np.array(pd_sla).size, np.array(pd_anno).size)
pd_anno = pd.Series(anno).reset_index()
plt.figure(figsize=(17, 13))
sns.heatmap(pd2_sla, annot=pd2_anno, fmt="", center=0, linewidths=.5)
plt.title('BBRFrcst vs BBRv2 Statistics')
plt.xlabel('Channel BW')
plt.ylabel('Channel RTT')


# %%

#%%
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#%%
def pckt_dict(path, parti, fnd):
    '''Extract pckts per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    patt = re.compile("I00(.*)[^I00]*"+fnd)
    d = dict()
    for i in patt.findall(text):
        d[int(i[:6])//parti] = d.setdefault(int(i[:6])//parti, 0) + 1
    return(d)

#%%
def byte_dict(path, parti, fnd):
    '''Extract bytes per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    patt = re.compile("I00(.*)[^I00]*"+fnd+"(.*)\n")
    d = dict()
    for i in patt.findall(text):
        d[int(i[0][:6])//parti] = d.setdefault(int(i[0][:6])//parti, 0) + int(re.search(" (\d*) bytes", i[1]).group(1))
    return(d)

#%%
# INPUT:
parti = 100         # Partition
statfile = "logsrv" # Name of file

d1 = pckt_dict(statfile, parti, "Sent packet:")
d2 = pckt_dict(statfile, parti, "Received packet:")

# %%
plt.figure(figsize=(12, 7))
plt.plot([parti*i for i in range(len(d1))], [d1.setdefault(i, 0) for i in range(len(d1))], label="Sent")
plt.plot([parti*i for i in range(len(d2))], [d2.setdefault(i, 0) for i in range(len(d2))], label="Received")
plt.xlabel('Timestamp')
plt.ylabel('Packets sent/recieved')
plt.title(f'Packets sent/recieved by [{statfile}] in timestamp I..')
plt.legend()
plt.show()

#%%
# INPUT:
parti = 50         # Partition
statfile = "logsrv" # Name of file

d1 = byte_dict(statfile, parti, "Sent packet:")
d2 = byte_dict(statfile, parti, "Received packet:")

# %%
plt.figure(figsize=(12, 7))
plt.plot([parti*i for i in range(len(d1))], [d1.setdefault(i, 0) for i in range(len(d1))], label="Sent")
plt.plot([parti*i for i in range(len(d2))], [d2.setdefault(i, 0) for i in range(len(d2))], label="Received")
plt.xlabel('Timestamp')
plt.ylabel('Bytes sent/recieved')
plt.title(f'Bytes sent/recieved by [{statfile}] in timestamp I..')
plt.legend()
plt.show()
# %%

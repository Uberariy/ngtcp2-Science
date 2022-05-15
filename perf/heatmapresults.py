"""
This program is an .ipybn notebook utility.

"""

#%%
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

#%%


#%%


#%%
# INPUT:
statfile = "perfres" # Name of file

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


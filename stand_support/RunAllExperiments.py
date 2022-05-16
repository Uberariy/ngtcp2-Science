import sys
import os
import time

if __name__ == "__main__":
    # Change file pathes below
    path_to_experiment = "experimentData.yml"
    path_to_change = "ChangeExperiment.py"
    path_to_mininet_run = "main.py"
    # Change parameters grid below
    rtts = [40, 80]
    bws = [80, 160]
    losses = [2]
    t_start = time.time()
    for i in rtts:
        for j in losses:
            for g in bws:
                t_one_start = time.time()
                print(f"RunAll: Running experiment - rtt: {i} loss: {j} bw: {g}\t... ")
                os.system(f"python3 {path_to_change} {path_to_experiment} {i} {j} {g}")
                os.system(f"sudo python3 /usr/bin/python3 {path_to_mininet_run} {path_to_experiment}")
                t_one_end = time.time()
                print(f"RunAll: Finished. Time taken: {t_one_start - t_one_end}")
    t_end = time.time()
    print(f"RunAll: Total time taken: {t_end - t_start}")

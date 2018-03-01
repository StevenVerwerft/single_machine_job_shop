import numpy as np
import pandas as pd
import sys
"""
This script will generate an instance for the single machine scheduling case.
"""


argv = sys.argv

try:
    amount_instances = int(argv[1])
except:
    amount_instances = 1


for instance_number in range(1, amount_instances+1):

    # make this dynamic
    # Amount of jobs
    n = 80
    id = range(1, n+1)

    # lower and upper bounds of individual job processing times
    p_l = 1
    p_u = 500

    # lower and upper bounds of individual fractions for setup times
    s_l = .25
    s_u = .75

    # Setup option to overcome to much independence between factors
    d = 1

    # Setup group fractions (fixed on three levels: Low, Medium, High)
    g = {"L": .1, "M": .3, "H": .9}
    nG = g["L"] * n  # this is the amount of different groups

    # Tardiness Factor (fixed on two levels: Low and High)
    t = {"L": .3, "H": .6}

    # Due date spread parameter
    w = {"L": .5, "H": 2.5}

    # Due date lower and upper bound => make this dynamic
    d_l = max(0, (1 - t["H"]) * (1 - w["H"]/2))
    d_u = min((1 - t["H"]) * (1 + w["H"]/2), 1)

    # set of processing times for each job
    P = np.random.uniform(p_l, p_u, n)

    # set of job group id's
    G = np.random.randint(1, nG+1, n)

    # fractions of job times to be used as setup times for each job
    S = np.random.uniform(s_l, s_u, n)

    # set of due date multipliers for the jobs
    D = np.random.uniform(d_l, d_u, n)

    # Makespan for the instance
    tot_p_time = sum(P)
    avg_s_time = np.mean(P*S)

    tot_avg_s = 0
    for i in range(len(P)):
        # average over all possible predecessors
        try:
            avg_s = np.append(S[:i]*P[:i], S[i+1:]*P[i+1:]).mean()
            tot_avg_s += avg_s
        except IndexError:
            # runs untill last job, but here index is out of bounds
            break
    tot_avg_s += np.mean(S[:-1]*P[:-1])

    ms = tot_p_time + (tot_avg_s * np.sqrt(g["L"]))
    print(ms)
    # instance dataframe
    instance = pd.DataFrame({'job_id': id, 'group_id': G,'p_times': P, 's_fracs': S, 'd_dates_multiplier': D, 'd_dates': D * ms})
    instance = instance[['job_id', 'group_id', 'p_times', 's_fracs', 'd_dates']]

    import os
    try:
        os.mkdir("instances")
    except:
        pass

    instance_path = 'instances/instance' + str(instance_number) + '.csv'
    instance.to_csv(instance_path, index=False)

quit()

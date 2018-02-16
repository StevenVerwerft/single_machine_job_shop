# imports
import numpy as np
import random
import pandas as pd
import itertools
import sys

# General options
stdout = sys.stdout
pd.set_option('display.width', 140)

# INSTANCE I/O
# Here comes the part where we read in the instance data
instance = pd.read_csv("instance.csv", sep=',')
instance['s_times'] = instance.s_fracs * instance.p_times
instance = instance[["job_id", "group_id", "p_times", "s_times", "d_dates"]]

n = instance.shape[0]
# Initialize tabu search: construct a valid solution. Use EDD rule and measure performance.
instance = instance.sort_values(by='d_dates')

# Calculate group process times, should be evaluated each iteration
g_sequence = np.ones(instance.shape[0])
g_sequence[1:] = np.where(instance.group_id.values[:-1] == instance.group_id.values[1:], 0, 1)
instance['g_sequence'] = g_sequence
instance['s_times'] = instance['s_times'] * instance['g_sequence']
instance['tot_time'] = instance.s_times + instance.p_times

# Calculate queue waiting time
tot_times = instance.p_times.values + instance.s_times.values
W = np.zeros(instance.shape[0])
W[1:] = np.cumsum(tot_times)[: -1]
instance['w_time'] = W

# Calculate lateness
L = instance.p_times.values + instance.s_times.values + instance.w_time.values - instance.d_dates.values
instance['lateness'] = np.maximum(np.zeros(len(L)), L)

print(max(instance.lateness.values))
Gvals = [max(instance.lateness.values)]
swaps = [(i, j) for i, j in itertools.combinations(range(instance.shape[0]), 2)]

# here comes the local search algorithm without tabulist

# memory that contains optimization paths
VALS = []
instance = instance.sample(80)
n_it = 500
for i in range(n_it):
    print('iteration', i)
    swap_vals = []
    local_optimum = True

    for swap in swaps:
        temp_instance = instance.copy()
        # moet lokaal gebeuren!
        temp_instance.loc[swap[0]], temp_instance.loc[swap[1]] = instance.loc[swap[1]], instance.loc[swap[0]]

        # Calculate group process times
        temp_g_sequence = np.ones(temp_instance.shape[0])
        temp_g_sequence[1:] = np.where(temp_instance.group_id.values[:-1] == temp_instance.group_id.values[1:], 0, 1)
        temp_instance['tot_time'] = temp_instance['s_times'] * temp_g_sequence + temp_instance['p_times']

        # Calculate queue waiting time
        temp_W = np.zeros(temp_instance.shape[0])
        temp_W[1:] = np.cumsum(temp_instance['tot_time'])[:-1]
        temp_instance['W_time'] = temp_W

        # Calculate lateness
        temp_l = temp_instance['p_times'].values + temp_instance['s_times'].values + temp_instance['W_time'].values \
                    - temp_instance['d_dates']
        temp_instance['lateness'] = np.maximum(np.zeros(len(temp_l)), temp_l)

        # Goal function: smallest maximal lateness
        gval = max(temp_instance.lateness.values)

        # use this when searching with best improving move
        # swap_vals.append(gval)

        # use this when searching with first improving move
        if gval < min(Gvals):
            Gvals.append(gval)
            print('better solution found! ', gval)
            print('Executing swap', swap)
            instance = temp_instance.copy()
            local_optimum = False
            break

    if local_optimum:
        print('local optimum, performing perturbation...')
        instance = instance.sample(instance.shape[0])
        # Calculate group process times, should be evaluated each iteration
        g_sequence = np.ones(instance.shape[0])
        g_sequence[1:] = np.where(instance.group_id.values[:-1] == instance.group_id.values[1:], 0, 1)
        instance['g_sequence'] = g_sequence
        instance['s_times'] = instance['s_times'] * instance['g_sequence']
        instance['tot_time'] = instance.s_times + instance.p_times

        # Calculate queue waiting time
        tot_times = instance.p_times.values + instance.s_times.values
        W = np.zeros(instance.shape[0])
        W[1:] = np.cumsum(tot_times)[: -1]
        instance['w_time'] = W

        # Calculate lateness
        L = instance.p_times.values + instance.s_times.values + instance.w_time.values - instance.d_dates.values
        instance['lateness'] = np.maximum(np.zeros(len(L)), L)
        print(instance.sort_values(by='lateness', ascending=False).head())
        VALS.append(Gvals)
        Gvals = [max(instance.lateness.values)]
        print('gvals = ', Gvals)


"""
This is used to give a 'new memory' to the search after a perturbation move is performed. 
Since it will not be possible to find a single improving move after performing the perturbation
"""
Gvals = [item for sublist in VALS for item in sublist]
Gvals = pd.DataFrame(Gvals, columns=['results'])
Gvals.to_csv('test.csv', sep=',')

quit()

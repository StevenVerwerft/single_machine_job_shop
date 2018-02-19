from functions import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tkinter import *

# Memory for found solutions
solution_memory = []

# input instance data
job_sequence = read_instance('instance.csv')

# initialize by applying EDD
job_sequence = EDD_rule(job_sequence)

# Evaluate solution
solution = evaluate_job_sequence(job_sequence)
solution_memory.append(solution)

# Generate move pool
swaps = create_swaps(job_sequence)
max_iter = 20
swap_mem = []

for i in range(max_iter):
    print("iteration", i)
    local_optimum = True
    for swap in swaps:
        swap_solution = evaluate_job_sequence(perform_swap(job_sequence, swap))
        swap_mem.append(swap_solution)
        if swap_solution < solution_memory[-1]:
            print('better solution found!')
            print('current solution quality:', swap_solution)
            solution_memory.append(swap_solution)
            job_sequence = perform_swap(job_sequence, swap)
            local_optimum = False
            break
    if local_optimum:
        print('local optimum')
        break

# calculate relative error with best found solution


fig, axes = plt.subplots(1, 3)
axes[0].plot(range(1, len(solution_memory)+1), solution_memory)
axes[0].set_xlabel('iterations')
axes[1].plot(range(1, len(swap_mem)+1), swap_mem)
axes[1].set_xlabel('swaps')
plt.ylabel("Goal function value")

axes[2].bar(np.arange(len(solution_memory)), (solution_memory - min(solution_memory))/min(solution_memory),
            align='center', alpha=.5)
axes[2].set_ylabel('error vs best')
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.show()



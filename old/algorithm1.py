from functions import *
import datetime
# fix for MACOSX GUI crash
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")  # use this backend to prevent macosX bug
import matplotlib.pyplot as plt  # should be put after bugfix
from matplotlib.ticker import FormatStrFormatter
from max_iter import max_iter, img_show, verbose

# Algorithm's name
name = 'det_ls'  # Deterministic Local Search Algorithm

# timestamp
ts = '{:%d%m_%H_%M_%s}'.format(datetime.datetime.now())

# io
max_iter = int(max_iter['max_iter'])

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

# Memory for goal function value per swap evaluation
swap_mem = []

# Local Search untill no local improvement found
for i in range(max_iter):
    print("iteration", i)
    local_optimum = True
    for swap in swaps:
        swap_solution = evaluate_job_sequence(perform_swap(job_sequence, swap))
        swap_mem.append(swap_solution)
        if swap_solution < solution_memory[-1]:
            if verbose:
                print('better solution found!')
            if verbose:
                print('current solution quality:', swap_solution)
            solution_memory.append(swap_solution)
            job_sequence = perform_swap(job_sequence, swap)
            local_optimum = False
            break
    if local_optimum:
        print('local optimum')
        break  # end algorithm when local optimum found

# Write solution path to external file

solution_path = 'solutions/solution'+ts+'_'+name
df = pd.DataFrame(solution_memory)
df.to_csv(solution_path, sep=',')

# Plot and save performance function
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].plot(range(1, len(solution_memory)+1), solution_memory)
axes[0].set_xlabel('iterations')
axes[0].set_title('Best Solution Found')
axes[0].set_ylabel("Goal function value")

axes[1].plot(range(1, len(swap_mem)+1), swap_mem)
axes[1].set_xlabel('swaps')
axes[1].set_title('Local Solution Found')
axes[2].bar(np.arange(len(solution_memory)), (solution_memory - min(solution_memory))/min(solution_memory),
            align='center', alpha=.5)
axes[2].set_title('error vs best (%)')
axes[2].set_xlabel('iterations')
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
filepath = 'img/fig'+ts+'.png'

plt.savefig(filepath)
os.chmod(filepath, 777)  # extra permissions for opening file automatically in darwin environment
if img_show:
    subprocess.Popen(['open ' + filepath], shell=True)

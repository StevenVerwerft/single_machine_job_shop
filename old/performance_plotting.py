import sys
import matplotlib.pyplot as plt
import pandas as pd
# Read in solutions from CLI
argv = sys.argv
solutions = [pd.read_csv(arg, sep=',', header=None, index_col=0, names=['solution']) for arg in argv if '.csv' in arg]

fig = plt.figure(figsize=(20, 10))

for sol in solutions:
    print(sol.solution.values)
    plt.plot(range(1, len(sol.solution.values)+1), sol.solution.values)

plt.show()
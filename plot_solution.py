import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


args = sys.argv

try:
    path = str(args[1])
except:
    print("first argument should be the path to the solution!")
    quit()

solution = pd.read_csv(path, index_col=0)

fig, ax = plt.subplots()
ax.plot(solution.timestamps, solution.goalvalues, label='label not yet implemented')
plt.xlabel('runtime (s)')
plt.ylabel('Goal function: minimal maximal tardiness')
plt.title('Local search for single machine job shop scheduling')
plt.legend()
plt.show()



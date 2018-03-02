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

try:
    for path in args[1:]:
        print(path)
        solution = pd.read_csv(path, index_col=0)

        # add a label to the plot
        label = 'No label found Dave...'
        names = path.split('/')  # split the path name in its subdirectories
        for name in names:
            if 'instance' in name:
                print()
                label = name

        # fig, ax = plt.subplots()
        plt.plot(solution.timestamps, solution.goalvalues, label=label)
        plt.xlabel('runtime (s)')
        plt.ylabel('Goal function: minimal maximal tardiness')
        plt.title('Local search for single machine job shop scheduling')
        plt.legend()

except:
    print('I can\'t let you do that Dave...')


plt.show()

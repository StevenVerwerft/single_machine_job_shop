import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('test.csv', index_col=0)
print(results)

results.plot()
plt.xlabel('iteration')
plt.ylabel('minimal maximal lateness')
plt.show()

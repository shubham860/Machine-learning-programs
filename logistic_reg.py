import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x =  np.arange(-10,10,0.01)
sig = 1/(1 + np.power(np.e, -x))
line = 4 * x + 7
sig_1 = (np.power(np.e, -x))/(1 + np.power(np.e, -x)) 

plt.plot(x,sig)
plt.show()


plt.plot(x,sig_1)
plt.show()


plt.plot(x,line)
plt.show()

sig_line = 1/(1 + np.power(np.e, -line))
plt.plot(x,sig_line)
plt.show()
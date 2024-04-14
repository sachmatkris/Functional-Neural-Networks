import numpy as np
from scipy.stats import loguniform
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
a, b = 1, 8
true_a, true_b = 0.00005, 0.05

loc = (true_b - b*true_a)/(b-a)
scale = (true_b + loc)/(b)

rv = loguniform(a, b)
x = np.linspace(a, b, 100)
ax.plot(x*scale - loc, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
fig.show()

(loc, scale)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms

# z = [0,0.1,0.3,0.9,1,2,5]
# z = list(range(0, 30000, 1000))
# with open('./ft_cat_epoch_ablation_for_drawing.txt', 'r') as f:
with open('./ft_cal_epoch_ablation_for_drawing_compose.txt', 'r') as f:
    epoch_results = f.readlines()
epoch_results = [i.strip().split(' ') for i in epoch_results]
epoch_results_array = np.array(epoch_results).astype(np.float)
z = [0,1,2,3,4,5,6,7,8,9,10,11,13,15,20,25,30,35]
# z = [0,1,2,3,4,5,6,7,8,9]

eAP = epoch_results_array[:, :4].mean(axis=1).tolist()
bin1 = epoch_results_array[:, 0].tolist()
bin2 = epoch_results_array[:, 1].tolist()
bin3 = epoch_results_array[:, 2].tolist()
bin4 = epoch_results_array[:, 3].tolist()

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)

matplotlib.rcParams.update({'font.size': 14})
ax1.plot(z, bin4, marker='o', linewidth=2,  label='AP of class bin [1000, -)')
ax1.plot(z, bin3, marker='o', linewidth=2,  label='AP of class bin [100, 1000)')
ax1.plot(z, bin2, marker='o', linewidth=2,  label='AP of class bin [10, 100)')
ax1.plot(z, bin1, marker='o', linewidth=2,  label='AP of class bin (0, 10)')

ax1.plot(z, eAP, linestyle='-', marker='o', linewidth=2,  label='bAP')


# ax1.plot([0],[15.4], 'D', color = 'green')

plt.xlabel('calibration steps (k)', size=16)
plt.ylabel('AP or bAP', size=16)
# ax1.set_xscale('log')

plt.legend( loc='best')

plt.grid()
plt.savefig('ablation_cal_steps.eps', format='eps', dpi=1000)
plt.show()



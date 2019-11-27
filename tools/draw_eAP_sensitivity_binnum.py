
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms


epoch_results = [[51, 250, 276, 253], [62, 278, 276, 214], [67, 298, 284, 181], [71, 325, 280, 154], [77, 348, 282, 123], [81, 362, 289, 98], [85, 377, 297, 71], [90, 386, 301, 53]]

epoch_results_array = np.array(epoch_results).astype(np.float)
z = [8,9,10,11,12,13,14,15]
# z = [0,1,2,3,4,5,6,7,8,9]

eAP = epoch_results_array[:, :4].mean(axis=1).tolist()
bin1 = epoch_results_array[:, 0].tolist()
bin2 = epoch_results_array[:, 1].tolist()
bin3 = epoch_results_array[:, 2].tolist()
bin4 = epoch_results_array[:, 3].tolist()

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)

matplotlib.rcParams.update({'font.size': 12})
ax1.plot(z, bin4, marker='o', linewidth=2,  label='class number in bin [f^3, -)')
ax1.plot(z, bin3, marker='o', linewidth=2,  label='class number in bin [f^2, f^3)')
ax1.plot(z, bin2, marker='o', linewidth=2,  label='class number in bin [f, f^2)')
ax1.plot(z, bin1, marker='o', linewidth=2,  label='class number in bin (0, f)')



# ax1.plot([0],[15.4], 'D', color = 'green')

plt.xlabel('calibration steps (k)', size=16)
plt.ylabel('AP or eAP', size=16)
# ax1.set_xscale('log')

plt.legend( loc='best')

plt.grid()
plt.savefig('eap_sensitivity_binnum.eps', format='eps', dpi=1000)
plt.show()




import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms

# z = [0,0.1,0.3,0.9,1,2,5]
z = [7.8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1230]
# thick = [20,40,20,60,37,32,21]ax1.set_xscale('log')
# thick=[15.4, 18.2, 18.7, 19.2, 19.4, 19.5, 19.9, 20.1, 20.4, 20.5, 20.6, 20.7, 20.8, 20.7, 20.7, 20.6, 20.6, 20.6, 20.5, 20.5, 19.8]
mrcnn=[15.4, 18.2, 18.7, 19.2, 19.4, 19.5, 19.9, 20.1, 20.4, 20.5, 20.6, 20.7, 20.8, 20.7, 20.7, 20.6, 20.6, 20.6, 20.5, 20.5, 19.8]

x_ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


# plt.plot([1.0],[44.8], 'D', color = 'black')
# plt.plot([0],[35.9], 'D', color = 'red')
# plt.plot([1.0],[56.8], 'D', color = 'black')

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(111)
matplotlib.rcParams.update({'font.size': 16})
ax1.plot(z, mrcnn, linestyle='dashed', marker='o', linewidth=2, c='k', label='mrcnn-r50-ag')
# ax1.plot(z, htc, marker='o', linewidth=2, c='g', label='htc')

ax1.plot([7.8],[15.4], 'D', color = 'green')
ax1.plot([1230],[19.8], 'D', color = 'red')


plt.xlabel('boundary threshold instance number T', size=16)
plt.ylabel('bAP', size=16)
# plt.gca().set_xscale('custom')
ax1.set_xscale('log')


ax1.set_xticks(x_ticks)
# from matplotlib.ticker import ScalarFormatter
# ax1.xaxis.set_major_formatter(ScalarFormatter())


plt.legend(['varying concatenation boundary','original head only', 'calibrated head only'], loc='best')

plt.minorticks_off()
plt.grid()
plt.savefig('concatenation_boundary.eps', format='eps', dpi=1000)
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# y1=[35.9, 43.4, 46.1, 49.3, 50.3, 51.3, 51.4, 49.9, 49.5, 48.5, 44.8]
# y2=[40.5, 48.2, 53.9 , 56.9, 57.8, 59.2, 58.3, 57.9, 57.5, 57.2, 56.8]
# y3=[61.5, 61.5, 61.5, 61.5, 61.5, 61.5, 61.5, 61.5, 61.5, 61.5, 61.5]
# x = np.linspace(0, 1, num=11, endpoint=True)
#
# f1 = interp1d(x, y1, kind='cubic')
# f2 = interp1d(x, y2, kind='cubic')
# f3 = interp1d(x, y3, kind='cubic')
# xnew = np.linspace(0, 1, num=101, endpoint=True)
# plt.plot(xnew, f3(xnew), '--', color='fuchsia')
# plt.plot(xnew, f1(xnew), '--', color='blue')
# plt.plot(xnew, f2(xnew), '--', color='green')
#
# plt.plot([0],[40.5], 'D', color = 'red')
# plt.plot([1.0],[44.8], 'D', color = 'black')
# plt.plot([0],[35.9], 'D', color = 'red')
# plt.plot([1.0],[56.8], 'D', color = 'black')
# plt.plot(x, y3, 'o', color = 'fuchsia')
# plt.plot(x, y1, 'o', color = 'blue')
# plt.plot(x, y2, 'o', color = 'green')
# plt.plot([0],[40.5], 'D', color = 'red')
# plt.plot([1.0],[44.8], 'D', color = 'black')
# plt.plot([0],[35.9], 'D', color = 'red')
# plt.plot([1.0],[56.8], 'D', color = 'black')
# plt.legend(['teacher','0.25x', '0.5x', 'full-feature-imitation', 'only GT supervison'], loc='best')
# plt.xlabel('Thresholding factor')
# plt.ylabel('mAP')
# plt.title('Resulting mAPs of varying thresholding factors')
# #plt.legend(['0.5x'])
# # plt.savefig('varying_thresh.eps', format='eps', dpi=1000)
# plt.show()

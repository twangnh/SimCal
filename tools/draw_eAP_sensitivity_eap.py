
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms


epoch_results_ours =  [0.2055814944259203, 0.2070789429222732, 0.21312734490544888, 0.21330987617112582, 0.21681729633603414, 0.21868618147095492, 0.22139531986572789, 0.2207302553866501]
epoch_results_imgsample = [0.20085911273955764, 0.20221658896725567, 0.2056222469806897, 0.2051128644774435, 0.208011478430485, 0.2117306883053619, 0.21469247380489614, 0.21509586079595858]
z = [8,9,10,11,12,13,14,15]

# fig = plt.figure(figsize=(8,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

matplotlib.rcParams.update({'font.size': 16})
ax1.plot(z, epoch_results_ours, marker='o', linewidth=2, color='darkorange',  label='r50-ours')
ax1.plot(z, epoch_results_imgsample, marker='o', linewidth=2, color='blue',  label='r50-IS')



# ax1.plot([0],[15.4], 'D', color = 'green')

plt.xlabel('bAP f value (m=3)', size=16)
plt.ylabel('bAP', size=16)
# ax1.set_xscale('log')

plt.legend( loc='best')

plt.grid()
plt.savefig('eap_sensitivity_eap_f.eps', format='eps', dpi=1000)
plt.show()

### eap m value
# [0.19348653350208908, 0.20481586368658788, 0.20629655179889703]
# [0.1960601492969575, 0.20901640943344768, 0.211437803122666]

epoch_results_imgsample = [0.1905114721878586, 0.2056222469806897, 0.2249353584074827]
epoch_results_ours = [0.19661301291002395, 0.21312734490544888, 0.2335877606226477]

z = [2,3,4]

fig = plt.figure()
ax1 = fig.add_subplot(111)

matplotlib.rcParams.update({'font.size': 16})
ax1.plot(z, epoch_results_ours, marker='o', linewidth=3, color='darkorange',  label='r50-ours')
ax1.plot(z, epoch_results_imgsample, marker='o', linewidth=3, color='blue',  label='r50-IS')

plt.xticks(np.arange(2, 5, step=1))

# ax1.plot([0],[15.4], 'D', color = 'green')

plt.xlabel('bAP m value (f=10)', size=16)
plt.ylabel('bAP', size=16)
# ax1.set_xscale('log')

plt.legend( loc='best')

plt.grid()
plt.savefig('eap_sensitivity_eap_m.eps', format='eps', dpi=1000)
plt.show()
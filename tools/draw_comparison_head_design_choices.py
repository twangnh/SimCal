import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['AP on bin (0,10)', 'AP on bin (10,100)']
baseline = [0.0, 13.3]
fc2_ncm = [6.0, 18.9]
fc2 = [8.6, 22.0]
fc3_rand = [9.1, 18.8]
fc3_ft = [13.2, 23.1]

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

matplotlib.rcParams.update({'font.size': 16})
# plt.rc('ytick', labelsize=10)

fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, baseline, width, label='baseline')
# rects2 = ax.bar(x - width/2, fc2_ncm, width, label='2fc_ncm')
# rects3 = ax.bar(x , baseline, fc2, label='baseline')
# rects4 = ax.bar(x + width/2, fc3_rand, width, label='2fc_ncm')
# rects5 = ax.bar(x + width, fc3_ft, width, label='baseline')

# Set position of bar on X axis
r1 = np.arange(len(labels))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]
r4 = [x + width for x in r3]
r5 = [x + width for x in r4]

# Make the plot
rects1 = ax.bar(r1, baseline, color='#7f6d5f', width=width, edgecolor='white', label='baseline')
rects2 = ax.bar(r2, fc2_ncm, color='#557f2d', width=width, edgecolor='white', label='2fc_ncm')
rects3 = ax.bar(r3, fc2,  width=width, edgecolor='white', label='2fc_rand')
rects4 = ax.bar(r4, fc3_rand,  width=width, edgecolor='white', label='3fc_rand')
rects5 = ax.bar(r5, fc3_ft,  width=width, edgecolor='white', label='3fc_ft')

ax.set_ylim([0,25])
ax.set_xticks([0.3, 1.3])
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()
plt.savefig('head_design_choices.eps', format='eps', dpi=1000)
plt.show()

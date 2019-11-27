import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Proposal Recall@1k', 'AP']
men_means = [55.9, 32.8]
women_means = [51.0, 18.0]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

matplotlib.rcParams.update({'font.size': 18})
plt.rc('ytick', labelsize=10)

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='COCO')
rects2 = ax.bar(x + width/2, women_means, width, label='LVIS')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_ylim([0,65])
ax.set_xticks(x)
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

fig.tight_layout()
plt.savefig('coco_lvis_pr_recall_bar.eps', format='eps', dpi=1000)
plt.show()

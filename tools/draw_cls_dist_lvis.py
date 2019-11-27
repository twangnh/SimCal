from matplotlib import pyplot as plt
import mmcv
import numpy as np
import pickle

train_info = pickle.load(open('./lvis_train_cats_info.pt', 'rb'))

y_lvis = pickle.load(open('./class_to_imageid_and_inscount.pt', 'rb'))
y_lvis = sorted([y_lvis[i]['isntance_count'] for i in range(1,1231)])[::-1]
# plt.figure(figsize=(90, 50))
# plt.bar(range(1, 1231), y_lvis, align='center', alpha=0.5, width=0.8)
plt.plot(range(1, 1231), y_lvis, color='black')
plt.fill_between(range(1, 480), 0, y_lvis[1:480], facecolor='green', interpolate=True)
plt.fill_between(range(481, 1230), 0, y_lvis[481:1230], facecolor='red')
# plt.fill_between(range(401, 500), 0, y_lvis[401:500])
# plt.fill_between(range(501, 1230), 0, y_lvis[501:1230])
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.yscale('log')
plt.ylabel('Number of training instances')
plt.xlabel('Sorted category index')
# plt.title('')
# plt.xticks(np.arange(len(x_name)), x_name, rotation=45)
plt.savefig('lvis_cls_dist.eps', format='eps', dpi=1000)
# plt.savefig('coco_sample_cls_dist_1.eps', format='eps')
plt.show()

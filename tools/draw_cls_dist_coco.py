from matplotlib import pyplot as plt
import mmcv
import numpy as np
import pickle

x_name = mmcv.load('./x_name.pkl')
y = mmcv.load('./y.pkl')

y_coco_sampled = pickle.load(open('./class_to_imageid_and_inscount_coco_sampled.pt', 'rb'))
y_coco_lt = sorted([y_coco_sampled[i]['isntance_count'] for i in range(1,81)])[::-1]
# plt.figure(figsize=(90, 50))
plt.bar(range(1, 81), y, align='center', alpha=0.5, width=0.8)
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.yscale('log')
plt.ylabel('Number of instances')
plt.xlabel('Sorted category index')
# plt.title('')
# plt.xticks(np.arange(len(x_name)), x_name, rotation=45)
plt.savefig('coco_orig_cls_dist.eps', format='eps')
# plt.savefig('coco_sample_cls_dist_1.eps', format='eps')
plt.show()

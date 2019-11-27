from lvis.lvis import LVIS
from mmdet.datasets.registry import DATASETS
import os.path as osp
import cv2
from matplotlib import pyplot as plt

@DATASETS.register_module
class LvisGtAnnVis():

    def __init__(self, ann_file):
        self.lvis = LVIS(ann_file)
        CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
                   'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                   'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
                   'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
                   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                   'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')##use CLASSES[self.cat2label[35]] to find the class name
        self.cat_ids = self.lvis.get_cat_ids()
        # self.cat2label = {
        #     cat_id: i + 1
        #     for i, cat_id in enumerate(self.cat_ids)
        # }
        self.img_ids = self.lvis.get_img_ids()
        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        self.img_infos = img_infos
        # self.img_prefix = './data/lvis/val2017'
        self.img_prefix = './data/lvis/train2017'

        self.filter_to_keep_finetune_classes()
    ###filter to keep only the finetune classes with zero_ap but not with fewshot training instances
    def filter_to_keep_finetune_classes(self):
        self.lvis._filter_anns_finetune_classes()## first filter anns
        keep_img_info_ids = []## then filter imgs
        for idx in range(len(self.img_infos)):
            img_id = self.img_infos[idx]['id']
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            for ann_id in ann_ids:
                if ann_id in self.lvis.anns.keys():
                    if idx not in keep_img_info_ids:
                        keep_img_info_ids.append(idx)
        self.img_infos = [self.img_infos[i] for i in keep_img_info_ids]

    def show(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ann_ids)
        # for ann in ann_info:
        #     if ann['iscrowd'] == 1:
        #         is_crowd_id_val.append(idx)
        if self.img_infos[idx]['filename'].startswith('COCO'):##for val set
            imdata = cv2.imread(osp.join(self.img_prefix, self.img_infos[idx]['filename'][13:]))
        else:
            imdata = cv2.imread(osp.join(self.img_prefix, self.img_infos[idx]['filename']))
        imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)
        plt.imshow(imdata)
        plt.axis('off')
        self.lvis.showanns(ann_info)

        plt.show()









if __name__ == '__main__':
    import pickle
    # ann_file = 'data/coco/annotations/instances_train2017.json'
    ann_file_train = 'data/lvis/lvis_v0.5_train.json'
    ann_file_val = 'data/lvis/lvis_v0.5_val.json'

    lvisGtVis_train = LvisGtAnnVis(ann_file_train)
    lvisGtVis_val = LvisGtAnnVis(ann_file_val)

    per_cat_info_train_ascending = sorted(lvisGtVis_train.lvis.dataset['categories'], key=lambda x: x['image_count']) #sort on ascending order
    per_cat_info_val_ascending = sorted(lvisGtVis_val.lvis.dataset['categories'], key=lambda x: x['image_count'])  # sort on ascending order

    per_cat_info_train_decending = sorted(lvisGtVis_train.lvis.dataset['categories'], key=lambda x: -x['image_count']) #sort on decending order
    per_cat_info_val_decending = sorted(lvisGtVis_val.lvis.dataset['categories'], key=lambda x: -x['image_count'])  # sort on decending order


    # pickle.dump(lvisGtVis_train.lvis.dataset['categories'], open('./lvis_train_cate_info.pt', 'wb'))
    # cate_info = pickle.load(open('./lvis_train_cate_info.pt', 'rb'))
    categories = lvisGtVis_val.lvis.dataset['categories']
    for idx in range(5000):
        lvisGtVis_val.show(idx)

    # is_crowd_id_val = pickle.load(open('./is_crowd_id_val.pt', 'rb'))
    # for idx in is_crowd_id_val:
    #     cocoGtVis.show(idx)

    # pickle.dump(is_crowd_id_val, open('./is_crowd_id_val.pt', 'wb'))

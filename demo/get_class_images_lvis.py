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

        valid_inds = self._filter_imgs()
        self.img_infos = [self.img_infos[i] for i in valid_inds]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.lvis.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

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

    ## for training data
    class_to_image={}
    for i in range(1231):
        class_to_image[i]={}
        class_to_image[i]['img_id'] = []
        class_to_image[i]['isntance_count'] = 0
        class_to_image[i]['image_info_id'] = []
        class_to_image[i]['category_id'] = i
    for idx in range(len(lvisGtVis_train.img_infos)):
        if idx%1000==0:
            print(idx)
        img_id = lvisGtVis_train.img_infos[idx]['id']
        ann_ids = lvisGtVis_train.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = lvisGtVis_train.lvis.load_anns(ann_ids)
        for item in ann_info:
            if item['image_id'] not in class_to_image[item['category_id']]['img_id']:
                class_to_image[item['category_id']]['img_id'].append(item['image_id'])
                class_to_image[item['category_id']]['image_info_id'].append(idx)
            class_to_image[item['category_id']]['isntance_count']+=1
    import pickle
    pickle.dump(class_to_image, open('./class_to_imageid_and_inscount.pt', 'wb'))
    print('df')

    ## for val data
    # class_to_image={}
    # for i in range(1231):
    #     class_to_image[i]={}
    #     class_to_image[i]['img_id'] = []
    #     class_to_image[i]['isntance_count'] = 0
    #     class_to_image[i]['image_info_id'] = []
    #     class_to_image[i]['category_id'] = i
    # for idx in range(len(lvisGtVis_val.img_infos)):
    #     if idx ==2988:
    #         print('')
    #     if idx%1000==0:
    #         print(idx)
    #     img_id = lvisGtVis_val.img_infos[idx]['id']
    #     ann_ids = lvisGtVis_val.lvis.get_ann_ids(img_ids=[img_id])
    #     ann_info = lvisGtVis_val.lvis.load_anns(ann_ids)
    #     for item in ann_info:
    #         if item['image_id'] not in class_to_image[item['category_id']]['img_id']:
    #             class_to_image[item['category_id']]['img_id'].append(item['image_id'])
    #             class_to_image[item['category_id']]['image_info_id'].append(idx)
    #         class_to_image[item['category_id']]['isntance_count']+=1
    # import pickle
    # pickle.dump(class_to_image, open('./class_to_imageid_and_inscount_val.pt', 'wb'))
    # print('df')




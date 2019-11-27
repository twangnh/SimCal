from pycocotools.coco import COCO
from mmdet.datasets.registry import DATASETS
import os.path as osp
import cv2
from matplotlib import pyplot as plt

@DATASETS.register_module
class CocoGtAnnVis():

    def __init__(self, ann_file):
        self.coco = COCO(ann_file)
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
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        self.img_infos = img_infos
        self.img_prefix = './data/coco/train2017'

    def show(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        # for ann in ann_info:
        #     if ann['iscrowd'] == 1:
        #         is_crowd_id_val.append(idx)
        imdata = cv2.imread(osp.join(self.img_prefix, self.img_infos[idx]['filename']))
        imdata = cv2.cvtColor(imdata, cv2.COLOR_BGR2RGB)
        plt.imshow(imdata)
        plt.axis('off')
        self.coco.showAnns(ann_info)

        plt.show()









if __name__ == '__main__':
    import pickle
    # ann_file = 'data/coco/annotations/instances_train2017.json'
    ann_file = 'data/coco/annotations/instances_train2017.json'

    cocoGtVis = CocoGtAnnVis(ann_file)
    cocoGtVis.show(14)
    for idx in range(5000):
        cocoGtVis.show(idx)

    # is_crowd_id_val = pickle.load(open('./is_crowd_id_val.pt', 'rb'))
    # for idx in is_crowd_id_val:
    #     cocoGtVis.show(idx)

    # pickle.dump(is_crowd_id_val, open('./is_crowd_id_val.pt', 'wb'))

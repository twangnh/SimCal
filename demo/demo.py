from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os

demopath = os.path.dirname(os.path.realpath(__file__))
# config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
config_file = 'configs/mask_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = os.path.join(demopath, '000000125100.jpg')  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

# test a list of images and write the results to image files
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs)):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     show_result(frame, result, model.CLASSES, wait_time=1)


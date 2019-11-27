from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector_normal, train_detector_calibration)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument('--use_model', help='use model')
    parser.add_argument('--exp_prefix', help='exp prefix')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    cfg.use_model = args.use_model
    cfg.exp_prefix = args.exp_prefix
     # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    ##coco train cls2imgsins statistics
    # class_to_image={}
    # for i in range(1,81):
    #     class_to_image[i]={}
    #     class_to_image[i]['img_id'] = []
    #     class_to_image[i]['isntance_count'] = 0
    #     class_to_image[i]['image_info_id'] = []
    #     class_to_image[i]['category_id'] = i
    # for idx in range(len(datasets[0].img_infos)):
    #     if idx%1000==0:
    #         print(idx)
    #     img_id = datasets[0].img_infos[idx]['id']
    #     ann_ids = datasets[0].coco.getAnnIds(imgIds=[img_id])
    #     ann_info = datasets[0].coco.loadAnns(ann_ids)
    #     for item in ann_info:
    #         label_id = datasets[0].cat2label[item['category_id']]
    #         if item['image_id'] not in class_to_image[label_id]['img_id']:
    #             class_to_image[label_id]['img_id'].append(item['image_id'])
    #             class_to_image[label_id]['image_info_id'].append(idx)
    #         class_to_image[label_id]['isntance_count']+=1
    # instancount = [class_to_image[i]['isntance_count'] for i in range(1, 81)]
    # import pickle
    # pickle.dump(class_to_image, open('./class_to_imageid_and_inscount_coco_sampled.pt', 'wb'))

    # if len(cfg.workflow) == 2:
    # datasets.append(build_dataset(cfg.data.train))#plain train
    # datasets.append(build_dataset(cfg.data.val))#val
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    if 'normal_training' in args.config:
        train_detector_normal(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger)
    else:
        train_detector_calibration(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            logger=logger)

if __name__ == '__main__':
    main()

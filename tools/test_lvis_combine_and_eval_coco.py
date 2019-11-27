import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval
from mmdet.core import lvis_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


import pickle
from collections import OrderedDict

def single_gpu_test(model, data_loader, olongtail_model, dataset_for_support, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, dataset_for_support=dataset_for_support,
                           dataset_val = data_loader.dataset, olongtail_model=olongtail_model, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].data.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'proposal_fast_percat', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--existing_json', type=str, help='existing_json')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = False

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    ## uncomment to only eval on first 100 imgs

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    print('load model from {}'.format(cfg.load_from))
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # set0 = mmcv.load('../liyu_mmdet/set0.pkl')
    # set1 = mmcv.load('../liyu_mmdet/set1.pkl')
    # set2 = mmcv.load('../liyu_mmdet/set2.pkl')
    # set3 = mmcv.load('../liyu_mmdet/set3.pkl')
    # set4 = mmcv.load('../liyu_mmdet/set4.pkl')
    # set5 = mmcv.load('../liyu_mmdet/set5.pkl')
    # set6 = mmcv.load('../liyu_mmdet/set6.pkl')
    # set7 = mmcv.load('../liyu_mmdet/set7.pkl')
    # set0 = mmcv.load('./set0.pkl')
    # set1 = mmcv.load('./set1.pkl')
    # set2 = mmcv.load('./set2.pkl')
    # set3 = mmcv.load('./set3.pkl')
    # set4 = mmcv.load('./set4.pkl')
    # set5 = mmcv.load('./set5.pkl')
    # set6 = mmcv.load('./set6.pkl')
    # set7 = mmcv.load('./set7.pkl')
    # set_combine = set0+set1+set2+set3+set4+set5+set6+set7
    # prefix = 'mrcnnr50_14.3_clshead'
    # set0 = mmcv.load('./{}_set0.pkl'.format(prefix))
    # set1 = mmcv.load('./{}_set1.pkl'.format(prefix))
    # set2 = mmcv.load('./{}_set2.pkl'.format(prefix))
    # set3 = mmcv.load('./{}_set3.pkl'.format(prefix))
    # set_combine = set0+set1+set2+set3

    # prefix = '/mrcnnr50_ag_coco_clshead'
    prefix = 'mrcnnr50_ag_3fc_ft_cocolongtail_cat400_epoch_2'
    prefix ='mrcnn_r50_ag_cocolt'
    print(prefix)

    set0 = mmcv.load('./{}_set0.pkl'.format(prefix))
    set1 = mmcv.load('./{}_set1.pkl'.format(prefix))
    set2 = mmcv.load('./{}_set2.pkl'.format(prefix))
    set3 = mmcv.load('./{}_set3.pkl'.format(prefix))
    set4 = mmcv.load('./{}_set4.pkl'.format(prefix))
    set5 = mmcv.load('./{}_set5.pkl'.format(prefix))
    set6 = mmcv.load('./{}_set6.pkl'.format(prefix))
    set7 = mmcv.load('./{}_set7.pkl'.format(prefix))

    # set0 = mmcv.load('./set0.pkl')
    # set1 = mmcv.load('./set1.pkl')
    # set2 = mmcv.load('./set2.pkl')
    # set3 = mmcv.load('./set3.pkl')
    # set4 = mmcv.load('./set4.pkl')
    # set5 = mmcv.load('./set5.pkl')
    # set6 = mmcv.load('./set6.pkl')
    # set7 = mmcv.load('./set7.pkl')
    set_combine = set0 + set1 + set2 + set3 + set4 + set5 + set6 + set7

    # set_liyu = mmcv.load('../mmdet_ensemble/results319.pkl')

    # mmcv.dump(set_combine, args.out)
    # result_files = results2json(dataset, set_combine,
    #                             args.out)
    print('pkl result dumped, start eval')
    # result_files = results2json(dataset, set_combine,
    #                             args.out, dump=False)
    #
    # lvis_eval(result_files, args.eval, dataset.lvis)


    result_files = results2json(dataset, set_combine, args.out, dump=False)
    coco_eval(result_files, args.eval, dataset.coco)

if __name__ == '__main__':
    main()

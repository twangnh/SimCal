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


from cls_head_models.simple2fc import simple2fc
from cls_head_models.simple3fc import simple3fc
import pickle
from collections import OrderedDict

def single_gpu_test(model, data_loader, cal_head, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, cal_head=cal_head, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'].data.size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, cal_head, show=False, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, cal_head=cal_head, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'].data.size(0)
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
    # parser.add_argument('checkpoint', help='checkpoint file')
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

    parser.add_argument(
        '--cal_head',
        default='3fc_ft',
        help='calibrated head for evaluation',
        type=str)

    parser.add_argument(
        '--head_ckpt',
        default='3fc_ft',
        help='checkpoint name of calibrated head',
        type=str)

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
    dataset.img_infos = dataset.img_infos[:20]

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
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    def load_ckpt(ncm_model, cal_head):
        print('load cls head {}'.format('{}/{}.pth'.format(cfg.work_dir, cal_head)))
        # epoch = torch.load('{}/{}_epoch.pth'.format(cfg.work_dir, cal_head))
        load_checkpoint(ncm_model, '{}/{}.pth'.format(cfg.work_dir, cal_head))
        # return epoch

    print('use {}'.format(args.cal_head))
    if len(dataset.CLASSES) == 1230:##lvis
        if '2fc_rand' in args.cal_head:
            calibrated_head = simple2fc().cuda()
        elif '3fc_rand' in args.cal_head or '3fc_ft' in args.cal_head:
            calibrated_head = simple3fc().cuda()
    elif len(dataset.CLASSES) ==80:## coco
        if '2fc_rand' in args.cal_head:
            calibrated_head = simple2fc(num_classes=81).cuda()
        elif '3fc_rand' in args.cal_head or '3fc_ft' in args.cal_head:
            calibrated_head = simple3fc(num_classes=81).cuda()
    # epoch = load_ckpt(calibrated_head, args.head_ckpt)
    load_ckpt(calibrated_head, args.head_ckpt)
    calibrated_head.eval()


    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, calibrated_head, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        calibrated_head = MMDistributedDataParallel(calibrated_head.cuda())
        outputs = multi_gpu_test(model, data_loader, calibrated_head, args.show, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        # mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if len(dataset.CLASSES) == 1230:
            if eval_types:

                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    lvis_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, args.out, dump=False)
                        print('Starting evaluate {}'.format(' and '.join(eval_types)))
                        lvis_eval(result_files, eval_types, dataset.lvis)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out + '.{}'.format(name)
                            result_files = results2json(dataset, outputs_,
                                                        result_file)
                            lvis_eval(result_files, eval_types, dataset.coco)
        elif len(dataset.CLASSES) == 80:
            result_files = results2json(dataset, outputs, args.out, dump=False)
            coco_eval(result_files, args.eval, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()

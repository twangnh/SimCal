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
from mmdet.core import lvis_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from cls_head_models.simple2fc import simple2fc
from cls_head_models.simple3fc import simple3fc

from mmcv.parallel import DataContainer as DC
import pickle

# class_to_imageid_and_inscount_val = pickle.load(open('./class_to_imageid_and_inscount_val_new.pt', 'rb'))
# fewshot_detected_id = [683, 246, 386, 358, 690, 155, 951, 218, 462, 790, 572, 300, 816, 569, 600, 845, 486, 836, 612, 70, 388, 1000, 117, 1030, 430, 1050, 767, 182, 843, 946, 1170, 541, 298, 127, 596, 423]
# {246: 0.7999999999999999, 358: 0.7999999999999999, 690: 0.7999999999999999, 155: 0.7999999999999999, 218: 0.2666666666666666, 572: 0.025, 569: 0.11428571428571424, 430: 0.18019801980198022, 182: 0.03366336633663366, 946: 0.40429042904290424, 1170: 0.24645964596459646}
# {'chest_of_drawers_(furniture)': 0.7999999999999999, 'dachshund': 0.7999999999999999, 'matchbox': 0.7999999999999999, 'broach': 0.7999999999999999, 'cargo_ship': 0.2666666666666666, 'hippopotamus': 0.025, 'heron': 0.11428571428571424, 'elk': 0.18019801980198022, 'cabin_car': 0.03366336633663366, 'seahorse': 0.40429042904290424, 'vulture': 0.24645964596459646}
# def single_gpu_test(model, data_loader, olongtail_model, dataset_for_support, show=False):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=not show, dataset_for_support=dataset_for_support,
#                            dataset_val = data_loader.dataset, olongtail_model=olongtail_model, **data)
#         results.append(result)
#
#         if show:
#             model.module.show_result(data, result, dataset.img_norm_cfg)
#
#         batch_size = data['img'][0].data.size(0)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results

#visialize fewshot classes
# def single_gpu_test(model, data_loader, olongtail_model, dataset_for_support, show=False):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     # for i, data in enumerate(data_loader):
#     #     if i%50==0:
#     #         print(i)
#     #     data_new = data
#     # interest_cat = fewshot_detected_id[0]
#
#     for interest_cat in fewshot_detected_id:
#         for i in class_to_imageid_and_inscount_val[interest_cat]['image_info_id']:
#             data = dataset.__getitem__(i)
#             data_new = {}
#             for key in data.keys():
#                 if key == 'img_meta' or key=='gt_masks':
#                     data_new[key] = DC([[data[key].data]], cpu_only=True)
#                 elif key == 'img':
#                     data_new[key] = DC([data[key].data.unsqueeze(0)])
#                 else:
#                     data_new[key] = DC([[data[key].data]])
#             with torch.no_grad():
#                 result = model(return_loss=False, rescale=not show, dataset_for_support=dataset_for_support,
#                                dataset_val = data_loader.dataset, olongtailmodel=olongtail_model, **data_new)
#             results.append(result)
#
#             if show:
#                 model.module.show_result(data_new, result, interest_cat=interest_cat)#data_new['img_meta'].data[0][0]['img_norm_cfg']
#                 # plt.imsave('./', img_show)
#             batch_size = data_new['img'].data[0].size(0)
#             for _ in range(batch_size):
#                 prog_bar.update()
#     return results,


def single_gpu_test(model, data_loader, olongtailmodel, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, olongtailmodel=olongtailmodel, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, olongtailmodel, show=False, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, olongtailmodel=olongtailmodel, **data)
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
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--set', type=int)
    parser.add_argument('--total_set_num', type=int)
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
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # per_set_img_num = int(len(dataset.img_infos)/args.total_set_num)
    # this_set_start = per_set_img_num*args.set
    # if args.set < args.total_set_num-1:
    #     dataset.img_infos = dataset.img_infos[this_set_start: this_set_start+per_set_img_num]
    # else:
    #     dataset.img_infos = dataset.img_infos[this_set_start:]
    # dataset.img_infos = dataset.img_infos[:100]

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # print('load from {}'.format(args.checkpoint))
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print('load model from {}'.format(cfg.load_from))
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


## load longtail classifier

    # def load_ncm_ckpt(ncm_model):
    #     if not os.path.exists('./simple3fc.pth'):
    #         print('start training from 0 epoch')
    #         return 0
    #     else:
    #         epoch = torch.load('./simple3fc_epoch.pth')
    #         load_checkpoint(ncm_model, './simple3fc.pth')
    #         return epoch

    # def load_ncm_ckpt(ncm_model):
    #     if not os.path.exists('./simple3fc.pth'):
    #         print('start training from 0 epoch')
    #         return 0
    #     else:
    #         epoch = torch.load('./finetune_simple3fc_epoch.pth')
    #         load_checkpoint(ncm_model, './finetune_simple3fc.pth')
    #         return epoch

    def load_ncm_ckpt(ncm_model):
        if not os.path.exists('./exp_randominit_negpossame_finetune_simple3fc_stage2_epoch.pth'):
            print('start training from 0 epoch')
            return 0
        else:
            epoch = torch.load('./exp_randominit_negpossame_finetune_simple3fc_stage2_epoch.pth')
            load_checkpoint(ncm_model, 'exp_randominit_negpossame_finetune_simple3fc_stage2.pth')
            return epoch

    # def load_simple2fc_stage0_ckpt(ncm_model):
    #
    #     epoch = torch.load('./finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage0_epoch.pth')
    #     load_checkpoint(ncm_model, './finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage0.pth')
    #     return epoch
    #
    # def load_simple2fc_stage1_ckpt(ncm_model):
    #
    #     epoch = torch.load('./finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage1_epoch.pth')
    #     load_checkpoint(ncm_model, './finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage1.pth')
    #     return epoch
    #
    # def load_simple2fc_stage2_ckpt(ncm_model):
    #
    #     epoch = torch.load('./finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage2_epoch.pth')
    #     load_checkpoint(ncm_model, './finetune2fc_10epoch/exp_randominit_finetune_simple2fc_stage2.pth')
    #     return epoch
    #
    #
    # olongtail_model_stage0 = simple2fc().cuda()
    # epoch = load_simple2fc_stage0_ckpt(olongtail_model_stage0)
    # print('load model epoch {}'.format(epoch))
    # olongtail_model_stage0.eval()
    #
    # olongtail_model_stage1 = simple2fc().cuda()
    # epoch = load_simple2fc_stage1_ckpt(olongtail_model_stage1)
    # olongtail_model_stage1.eval()
    #
    # olongtail_model_stage2 = simple2fc().cuda()
    # epoch = load_simple2fc_stage2_ckpt(olongtail_model_stage2)
    # olongtail_model_stage2.eval()
    #
    # olongtail_model_all_stage = [olongtail_model_stage0, olongtail_model_stage1, olongtail_model_stage2]

    prefix = '3fc_ft'
    def load_stage0_ckpt(ncm_model):

        # epoch = torch.load('./work_dirs/htc/{}_stage0_epoch.pth'.format(prefix))
        load_checkpoint(ncm_model, './work_dirs/htc/{}_stage0.pth'.format(prefix))
        # return epoch

    def load_stage1_ckpt(ncm_model):

        # epoch = torch.load('./work_dirs/htc/{}_stage1_epoch.pth'.format(prefix))
        load_checkpoint(ncm_model, './work_dirs/htc/{}_stage1.pth'.format(prefix))
        # return epoch

    def load_stage2_ckpt(ncm_model):

        # epoch = torch.load('./work_dirs/htc/{}_stage2_epoch.pth'.format(prefix))
        load_checkpoint(ncm_model, './work_dirs/htc/{}_stage2.pth'.format(prefix))
        # return epoch


    olongtail_model_stage0 = simple3fc().cuda()
    epoch = load_stage0_ckpt(olongtail_model_stage0)
    # print('load model epoch {}'.format(epoch))
    olongtail_model_stage0.eval()

    olongtail_model_stage1 = simple3fc().cuda()
    epoch = load_stage1_ckpt(olongtail_model_stage1)
    olongtail_model_stage1.eval()

    olongtail_model_stage2 = simple3fc().cuda()
    epoch = load_stage2_ckpt(olongtail_model_stage2)
    olongtail_model_stage2.eval()

    olongtail_model_all_stage = [olongtail_model_stage0, olongtail_model_stage1, olongtail_model_stage2]


    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, olongtail_model_all_stage, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, olongtail_model_all_stage, args.show, args.tmpdir)

    # mmcv.dump(outputs, args.out)
    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        # mmcv.dump(outputs, args.out)
        eval_types = args.eval
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

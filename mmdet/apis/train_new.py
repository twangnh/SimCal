from __future__ import division
import re
from collections import OrderedDict

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
# from protonets_models.runner_protonet_training import Runner_Proto

from mmdet import datasets
from mmdet.core import (CocoDistEvalmAPHook, CocoDistEvalRecallHook,
                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.models import RPN
from .env import get_root_logger
import torch.nn.functional as F
from mmcv.runner import get_dist_info, load_checkpoint, save_checkpoint


from cls_head_models.simple2fc import simple2fc
from cls_head_models.simple3fc import simple3fc
from cls_head_models.utils import *

import torch.optim as optim
import os





def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            dataset_type = DATASETS.get(val_dataset_cfg.type)
            if issubclass(dataset_type, datasets.CocoDataset):
                runner.register_hook(
                    CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))
            else:
                runner.register_hook(
                    DistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def compose_training_data(neg_feats, gt_inds, gt_num, pos_feats, pos_label, data_cls_index, bs, nc):

    gt_num_cum = torch.cat([torch.zeros(1).long().cuda(), torch.cumsum(gt_num, 0)])
    pos_label_split_per_img = []
    pos_feat_split_per_img = []
    for i in range(gt_num.size(0)):  ## iterate over whole batch
        pos_label_split_per_img.append(pos_label[gt_num_cum[i]:gt_num_cum[i + 1]])
        pos_feat_split_per_img.append(pos_feats[gt_num_cum[i]:gt_num_cum[i + 1]])

    samples = []
    for nc_i in range(nc):
        for per_cls_im_i in range(int(bs / nc)):
            index = pos_label_split_per_img[nc_i * 1 + per_cls_im_i] == data_cls_index[
                nc_i * 1 + per_cls_im_i]
            temp = pos_feat_split_per_img[nc_i * 1 + per_cls_im_i][index]
            # if temp.size(0)>5:## uncomment to subsample to 5
            #     temp = temp[torch.randperm(temp.size(0))[:5]]
            samples.append(temp)

    pos_num = torch.cat(samples).size(0)

    ## sampel bg to be the same as fg
    neg_feats_sampled = neg_feats[torch.randperm(neg_feats.size(0))[:pos_num]]
    samples = [neg_feats_sampled] + samples
    data_cls_index = np.concatenate([[0], data_cls_index])  ## neg label

    label_converts = []
    for i, item in enumerate(samples):
        label_converts.append(item.new_zeros(item.size(0)).fill_(data_cls_index[i]))
    label_converts = torch.cat(label_converts).long()
    samples_convert = torch.cat(samples)

    return samples_convert, label_converts


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = []
    # for idx, ds in enumerate(dataset):
    #     if idx==0:
    #         data_loaders.append(build_dataloader(
    #             ds,
    #             cfg.data.imgs_per_gpu,
    #             cfg.data.workers_per_gpu,
    #             cfg.gpus,
    #             dist=False,
    #             cls_balanced_sampler=False))
    #     else:
    #         data_loaders.append(build_dataloader(
    #             ds,
    #             cfg.data.imgs_per_gpu,
    #             cfg.data.workers_per_gpu,
    #             cfg.gpus,
    #             dist=False,
    #             cls_balanced_sampler=True))
    data_loaders.append(build_dataloader(
        dataset[0],
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        cfg.gpus,
        dist=False,
        cls_balanced_sampler=True))

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    load_checkpoint(model, cfg.load_from)
    print('load from {}'.format(cfg.load_from))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


    # for param in model.parameters():
    #     param.requires_grad = False
    # model.eval()

    dataset_to_longtailmodel={}
    dataset_to_longtailmodel['train'] = data_loaders[0]
    # dataset_to_longtailmodel['train_plain'] = data_loaders[1]
    # dataset_to_longtailmodel['val'] = data_loaders[2]

## which to use
    import datetime
    # cls_models = {
    #               1: '2fc_rand',
    #               2: '3fc_rand',
    #               3: '3fc_ft' }

    use_model = cfg.use_model
    print('use {}'.format(cfg.use_model))
    exp_prefix = cfg.exp_prefix
    total_epoch = 120
    initial_lr= 0.01


    if hasattr(dataset[0], 'coco'):
        if use_model == '2fc_rand':
            cls_head = simple2fc(num_classes=81).cuda()
        elif use_model == '3fc_rand' or '3fc_ft':
            cls_head = simple3fc(num_classes=81).cuda()

    elif hasattr(dataset[0], 'lvis'):
        if use_model == '2fc_rand':
            cls_head = simple2fc(num_classes=1231).cuda()
        elif use_model == '3fc_rand' or '3fc_ft':
            cls_head = simple3fc(num_classes=1231).cuda()

    optimizer = optim.SGD([{'params': cls_head.parameters(),
                            'lr': initial_lr}])

    # for param in list(cls_head.parameters())[:-4]:
    #     param.requires_grad = False

    def save_ckpt(cls_head):
        save_checkpoint(cls_head, './{}/{}_{}.pth'.format(cfg.work_dir, exp_prefix, use_model))
        torch.save(epoch, './{}/{}_{}_epoch.pth'.format(cfg.work_dir, exp_prefix, use_model))

    def load_ckpt(cls_head, use_model):

        if use_model == '2fc_rand' or use_model == '3fc_rand':
            if not os.path.exists('./{}_{}.pth'.format(exp_prefix, use_model)):
                print('start training from 0 epoch')
                return 0
            else:
                epoch = torch.load('./{}_{}_epoch.pth'.format(exp_prefix, use_model))
                load_checkpoint(cls_head, './{}_{}.pth'.format(exp_prefix, use_model))
                return epoch

        elif use_model == '3fc_ft':
            if not os.path.exists('./{}_{}.pth'.format(exp_prefix, use_model)):
                print('start training from 0 epoch, init from orig 3fc cls head')

                orig_head_state_dict = model.module.bbox_head.state_dict()
                key_map = {'fc_cls.weight': 'feat_classifier.fc_classifier.weight',
                           'fc_cls.bias': 'feat_classifier.fc_classifier.bias',
                           'shared_fcs.0.weight': 'feat_classifier.fc1.weight',
                           'shared_fcs.0.bias': 'feat_classifier.fc1.bias',
                           'shared_fcs.1.weight': 'feat_classifier.fc2.weight',
                           'shared_fcs.1.bias': 'feat_classifier.fc2.bias'}
                new_state_dict = OrderedDict()

                for key, value in orig_head_state_dict.items():
                    if key in key_map:
                        new_key = key_map[key]
                        new_state_dict[new_key] = value

                cls_head.load_state_dict(new_state_dict)
                return 0
            else:
                epoch = torch.load('./{}_{}_epoch.pth'.format(exp_prefix, use_model))
                load_checkpoint(cls_head, './{}_{}.pth'.format(exp_prefix, use_model))
                return epoch

    epoch = load_ckpt(cls_head, use_model)
    cls_head = MMDataParallel(cls_head, device_ids=range(cfg.gpus)).cuda()


    for epoch in range(epoch + 1, total_epoch+1):

        ##due to schdualer bug, we do manual lr schedule
        if epoch >= 8:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr*0.1
        if epoch >= 11:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr*0.01
        print('epoch {} lr {}'.format(epoch, optimizer.param_groups[0]['lr']))

        for step, data_batch in enumerate(dataset_to_longtailmodel['train']):
            if step % 10 == 0:
                print('step {} time: {}'.format(step, datetime.datetime.now()))
                torch.cuda.empty_cache()

            bs = dataset_to_longtailmodel['train'].batch_size
            nc = dataset_to_longtailmodel['train'].sampler.nc
            data_cls_index = np.split(dataset_to_longtailmodel['train'].sampler.class_indices,
                                      dataset_to_longtailmodel['train'].sampler.episode, 0)[step]

            neg_feats, gt_inds, gt_num, pos_feats, pos_label = model(**data_batch)
            samples_convert, label_converts = compose_training_data(neg_feats, gt_inds, gt_num, pos_feats, pos_label
                                                                    , data_cls_index, bs, nc)

            logits = cls_head(samples_convert)

            log_p_y = F.log_softmax(logits, dim=1).view(logits.size(0), -1)

            loss_target = -log_p_y.gather(1, label_converts.unsqueeze(1)).squeeze().view(-1).mean()

            loss_val = 1.0 * loss_target

            _, y_hat = log_p_y.max(1)
            acc_val = torch.eq(y_hat, label_converts).float().mean()
            bg_acc = (y_hat[label_converts == 0] == 0).sum() / (label_converts == 0).sum().float()
            fg_acc = torch.eq(y_hat, label_converts).float()[label_converts != 0].mean()


            if step % 10 == 0:
                print('step {} acc: {}'.format(step, acc_val.item()))
                print('step {} bg acc: {}'.format(step, bg_acc.item()))
                print('step {} fg acc: {}'.format(step, fg_acc.item()))

            optimizer.zero_grad()
            loss_val.backward()

            optimizer.step()

        save_ckpt(cls_head)



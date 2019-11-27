from __future__ import division

import math
import torch
import numpy as np

from mmcv.runner.utils import get_dist_info
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler
import pickle
import random

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class EpisodicSampler(Sampler):

    def __init__(self, dataset, batch_size_total, nc, episode):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        # self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        # for i, size in enumerate(self.group_sizes):
        #     self.num_samples += int(np.ceil(
        #         size / self.samples_per_gpu)) * self.samples_per_gpu

        ## for coco
        if hasattr(dataset, 'coco'):
            self.dataset_class_image_info = pickle.load(open('./class_to_imageid_and_inscount_coco_sampled.pt', 'rb'))
            self.dataset_class_image_info_tolist = [self.dataset_class_image_info[cls_idx]
                                                      for cls_idx in range(1, len(self.dataset_class_image_info) + 1)
                                                      if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        elif hasattr(dataset, 'lvis'):
            self.dataset_class_image_info = pickle.load(open('./class_to_imageid_and_inscount.pt', 'rb'))
            self.dataset_class_image_info_tolist = [self.dataset_class_image_info[cls_idx]
                                                      for cls_idx in range(len(self.dataset_class_image_info))
                                                      if self.dataset_class_image_info[cls_idx]['isntance_count'] > 0]
        # self.dataset_abundant_class_image_info = [self.dataset_class_image_info[cls_idx]
        #                                           for cls_idx in range(len(self.dataset_class_image_info))
        #                                           if self.dataset_class_image_info[cls_idx]['isntance_count'] > 1000]

        self.dataset_abundant_class_ids = [item['category_id'] for item in self.dataset_class_image_info_tolist]
        self.nc = nc
        self.bs = batch_size_total
        self.episode = episode
    def __iter__(self):

        ##first, sample per episode classes(NC) self.bs/self.nc is the num of img per class
        indices = []
        class_indices = []

        #tau =0 corresponds to class-balanced sampling, larger value has bias for many-shot classes
        tau = 0.0
        # print('tau value {}'.format(tau))
        for i in range(self.episode):## possiblly need to ensure per gpu images are with similar ratio
            cls_ins_count = [item['isntance_count'] for item in self.dataset_class_image_info_tolist]
            cls_ins_count_thr = [i if i < 100 else 100 for i in cls_ins_count]
            cls_ins_count_thr_log = np.array(cls_ins_count_thr)**tau
            cls_probs = cls_ins_count_thr_log / sum(cls_ins_count_thr_log)

            ##sample with probability
            per_episode_cls_sampled = np.random.choice(self.dataset_class_image_info_tolist, self.nc, p=cls_probs).tolist()

            ##banlanced sample
            # per_episode_cls_sampled = random.sample(self.dataset_class_image_info_tolist, self.nc)

            # per_episode_image_sampled = [random.sample(item['image_info_id'], int(self.bs/self.nc)) for item in per_episode_cls_sampled]
            per_episode_image_sampled = [random.choices(item['image_info_id'], k=int(self.bs / self.nc)) for item in
                                         per_episode_cls_sampled]
            indices.append(np.concatenate(per_episode_image_sampled))
            class_indices.append(np.stack([item['category_id'] for item in per_episode_cls_sampled for i in range(int(self.bs/self.nc))]))
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        self.class_indices = np.concatenate(class_indices)## for external access of current nc classes, this is an ugly way out of dataloader
        assert len(indices) == self.episode*self.bs
        return iter(indices)

    def __len__(self):
        return self.episode*self.bs



class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

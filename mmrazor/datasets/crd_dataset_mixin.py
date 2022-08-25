# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

import numpy as np

from mmrazor.registry import DATASETS


class CRD_ClsDatasetMixin(object):
    """The reconstructed dataset for crd distillation reconstructed dataset
    adds another four parameters to control the resampling of data. If
    is_train, the dataset will return one positive sample and index of neg_num
    negative sample; else the dataset only return one positive sample like the
    dataset in mmcls library.

    Args:
        is_train: If True, `` __getitem__`` method will return the positive
            sample and index of `neg_num` negative sample, usually used in
            the training of crd distillation. And 'False' is used in the
            eval and test dataset of crd distillation.
        mode: Controls how negative samples are generated.
        percent: Control the cutting ratio of negative samples.
        neg_num: The index length of negative samples.

    Returns:
        dict: Dict of dataset sample. The following fields are contained.
            - img and gt_label: Same as the dataset in mmcls library.
            - contrast_sample_idx: the indexes of contrasted
                samples(neg_num + 1).
            - idx: The index of sample.
    """
    num_classes = 10

    def __init__(self,
                 dataset: Dict,
                 neg_num: int = 16384,
                 sample_mode: str = 'exact',
                 percent: float = 1.0) -> None:
        serialize_data = dataset.get('serialize_data', False)
        if serialize_data:
            raise NotImplementedError(
                '`serialize_data` is not supported for now.')
        dataset['serialize_data'] = serialize_data
        self.dataset = DATASETS.build(dataset)

        self._parse_fullset_contrast_info(neg_num, sample_mode, percent)

    def _parse_fullset_contrast_info(self, neg_num: int, sample_mode: str,
                                     percent: float) -> None:
        # parse contrast info.
        assert sample_mode in ['exact', 'random']
        if not self.dataset.test_mode:
            # Must fully initialize dataset first.
            self.dataset.full_init()

            # Parse info.
            self.gt_labels = [
                data['gt_label'] for data in self.dataset.data_list
            ]
            self.neg_num = neg_num
            self.sample_mode = sample_mode
            self.num_samples = self.dataset.__len__()

            self.cls_positive: List[List[int]] = [
                [] for i in range(self.dataset.num_classes)
            ]
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [
                [] for i in range(self.dataset.num_classes)
            ]
            for i in range(self.dataset.num_classes):
                for j in range(self.dataset.num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i])
                for i in range(self.dataset.num_classes)
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i])
                for i in range(self.dataset.num_classes)
            ]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(self.dataset.num_classes)
                ]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def _get_contrast_info(self, data, idx):
        if self.sample_mode == 'exact':
            pos_idx = idx
        elif self.sample_mode == 'random':
            pos_idx = np.random.choice(self.cls_positive[self.gt_labels[idx]],
                                       1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.sample_mode)
        replace = True if self.neg_num > \
            len(self.cls_negative[self.gt_labels[idx]]) else False
        neg_idx = np.random.choice(
            self.cls_negative[self.gt_labels[idx]],
            self.neg_num,
            replace=replace)
        contrast_sample_idxs = np.hstack((np.asarray([pos_idx]), neg_idx))
        data['contrast_sample_idxs'] = contrast_sample_idxs
        return data

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.
        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.dataset.get_data_info(idx)
        if not self.dataset.test_mode:
            data_info = self._get_contrast_info(data_info, idx)
        return self.dataset.pipeline(data_info)

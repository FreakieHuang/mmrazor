# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

import numpy as np

from mmrazor.registry import DATASETS


@DATASETS.register_module()
class CRD_ClsDataset(object):
    """Dataset wrapper for CRD algorithm on classification datasets.

    Args:
        dataset (Dict): Original dataset dict.
        neg_num (int, optional): Number of negative samples. Defaults to 16384.
        sample_mode (str, optional): Sample mode. Defaults to 'exact'.
        percent (float, optional): Sampling percentage. Defaults to 1.0.
    """

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
        """parse contrast information of the whole dataset.

        Args:
            neg_num (int): negative sample number.
            sample_mode (str): sample mode.
            percent (float): sampling percentage.
        """
        assert sample_mode in ['exact', 'random']
        if not self.dataset.test_mode:
            # Must fully initialize dataset first.
            self.dataset.full_init()
            self.num_classes: int = len(self.dataset.CLASSES)

            # Parse info.
            self.gt_labels = [
                data['gt_label'] for data in self.dataset.data_list
            ]
            self.neg_num = neg_num
            self.sample_mode = sample_mode
            self.num_samples = self.dataset.__len__()

            self.cls_positive: List[List[int]] = [
                [] for i in range(self.num_classes)
            ]
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [
                [] for i in range(self.num_classes)
            ]
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i])
                for i in range(self.num_classes)
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i])
                for i in range(self.num_classes)
            ]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(self.num_classes)
                ]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def _get_contrast_info(self, data: Dict, idx: int):
        """Get contrast information for each data sample."""
        if self.sample_mode == 'exact':
            pos_idx = idx
        elif self.sample_mode == 'random':
            pos_idx = np.random.choice(self.cls_positive[self.gt_labels[idx]],
                                       1)
            pos_idx = pos_idx[0]  # type: ignore
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

    @property
    def img_prefix(self):
        """The prefix of images."""
        return self.dataset.img_prefix

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self.dataset.CLASSES

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return self.dataset.class_to_idx

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        return self.dataset.get_gt_labels()

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return self.dataset.get_cat_ids(idx)

    def _compat_classes(self, metainfo, classes):
        """Merge the old style ``classes`` arguments to ``metainfo``."""
        return self.dataset._compat_classes(metainfo, classes)

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True."""
        self.dataset.full_init()

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__ + '\n'
        return head + self.dataset.__repr__()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        return self.dataset.get

# Copyright (c) OpenMMLab. All rights reserved.
from operator import attrgetter

import mmcv
import torch.nn as nn
from mmcv.runner import Hook

from mmrazor.registry import HOOKS


class LossWeightUpdaterHook(Hook):
    """Update Loss weight for a Loss module before an epoch or iter
    Args:
        loss (str | nn.module):  a loss object (etc., L1Loss())
                                or the attribute name of a loss object
                                (etc., 'layer.head.loss')
        by_epoch (bool, optional):  Loss weight changes epoch by epoch.
                                    Defaults to True.
    """

    def __init__(self, loss, by_epoch=True):
        self.loss = loss
        self.loss_module = None
        self.base_lw = None
        self.regular_lw = None
        self.by_epoch = by_epoch

    def before_run(self, runner):
        if isinstance(self.loss, str):
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            self.loss_module = attrgetter(self.loss)(model)
        elif isinstance(self.loss, nn.Module):
            self.loss_module = self.loss
        elif self.loss is None:
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            self.loss_module = model

        assert hasattr(self.loss_module, 'loss_weight'), (
            f'module ({self.loss_module}) has no attribute `loss_weight`')
        self.base_lw = self.loss_module.loss_weight
        self.loss = None
        del self.loss

    def _set_lw(self, lw: float):
        self.loss_module.loss_weight = lw

    def get_lw(self, runner, base_lw):
        raise NotImplementedError

    def before_train_epoch(self, runner):
        if self.by_epoch:
            new_lw = self.get_lw(runner, self.base_lw)
            self._set_lw(new_lw)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            new_lw = self.get_lw(runner, self.base_lw)
            self._set_lw(new_lw)


@HOOKS.register_module()
class StepLwUpdaterHook(LossWeightUpdaterHook):
    """Step Loss weight scheduler with min_alpha clipping.

    Args:
        step (int | list[int]): Step to decay the loss weight.
        gamma (float, optional): Decay alpha ratio. Default: 0.1.
        min_alpha (float, optional): Minimum alpha value to keep.
    """

    def __init__(self, loss, by_epoch, step, gamma=0.1, min_lw=None):

        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all(s > 0 for s in step)
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lw = min_lw
        super(StepLwUpdaterHook, self).__init__(loss, by_epoch)

    def get_lw(self, runner, base_lw):

        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = next(
                (i for i, s in enumerate(self.step) if progress < s),
                len(self.step),
            )

        lw = base_lw * (self.gamma**exp)
        if self.min_lw is not None:
            # clip to a minimum value
            lw = max(lw, self.min_lw)
        return lw


@HOOKS.register_module()
class ExpLwUpdaterHook(LossWeightUpdaterHook):
    """Exponential loss weight updater This schedule applies an exponential
    decay function to loss weight."""

    def __init__(self, loss, by_epoch, gamma):
        self.gamma = gamma
        super(ExpLwUpdaterHook, self).__init__(loss, by_epoch)

    def get_lw(self, runner, base_lw):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lw * self.gamma**progress


@HOOKS.register_module()
class PolyLwUpdaterHook(LossWeightUpdaterHook):
    """Polynomial loss weight updater Update loss weight in PolynomialDecay."""

    def __init__(self, loss, by_epoch, power=1.0, min_lw=0.0):
        self.power = power
        self.min_lw = min_lw
        super(PolyLwUpdaterHook, self).__init__(loss, by_epoch)

    def get_lw(self, runner, base_lw):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lw - self.min_lw) * coeff + self.min_lw


@HOOKS.register_module()
class InvLwUpdaterHook(LossWeightUpdaterHook):
    """Inverse powerd loss weight update, decaying as:

    base_alpha * (1 + gamma * iter/epoch) ^ (- power)
    """

    def __init__(self, loss, by_epoch, gamma, power=1.0):
        self.gamma = gamma
        self.power = power
        super(InvLwUpdaterHook, self).__init__(loss, by_epoch)

    def get_lw(self, runner, base_lw):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lw * (1 + self.gamma * progress)**(-self.power)

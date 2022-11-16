# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch
import torch.nn
from mmcls.structures import ClsDataSample
from mmengine.config import Config, DictAction

from mmrazor.fx.graph_module import MMGraphModule, MMObservedGraphModule
from mmrazor.models import FXModelWrapper
from mmrazor.registry import MODELS
from mmrazor.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train an algorithm')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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


def custom_trace(
    root,
    quantizer,
    concrete_args=None,
):
    root.mode = 'loss'  # type: ignore
    graph_loss = quantizer.tracer.trace(root, concrete_args)
    root.mode = 'predict'  # type: ignore
    graph_predict = quantizer.tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__
        if isinstance(root, torch.nn.Module) else root.__name__)
    graphs = dict(loss=graph_loss, predict=graph_predict)
    return MMGraphModule(quantizer.tracer.root, graphs, name)


def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build model
    ori_model = MODELS.build(cfg.model.architecture)
    quantizer = MODELS.build(cfg.model.quantizer)
    quantizer._swap_ff_with_fxff(ori_model)

    fx_wrapper = FXModelWrapper(ori_model)
    graph_module: MMGraphModule = custom_trace(fx_wrapper, quantizer)
    print('ori loss graph: ', graph_module)
    graph_module.to_mode('predict')
    print('ori predict graph: ', graph_module)

    graph_module: MMObservedGraphModule = quantizer.prepare(
        ori_model, graph_module)
    print('loss graph: ', graph_module)
    graph_module.to_mode('predict')
    print('predict graph: ', graph_module)

    # test forward
    inputs = torch.rand(1, 3, 224, 224)
    data_samples = [ClsDataSample().set_gt_label(1)]

    # loss
    graph_module.to_mode('loss')
    print('loss: ', graph_module(inputs, data_samples))
    graph_module.mode = 'predict'
    print('predict: ', graph_module(inputs, data_samples))


if __name__ == '__main__':
    main()

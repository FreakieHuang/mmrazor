# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List

import torch
import torch.nn
from torch.ao.quantization import propagate_qconfig_, swap_module
from torch.ao.quantization.qconfig_dict_utils import get_flattened_qconfig_dict
from torch.ao.quantization.quantization_mappings import (
    get_default_qat_module_mappings, get_default_static_quant_module_mappings)
from torch.ao.quantization.quantize_fx import (_convert_fx, _fuse_fx,
                                               _swap_ff_with_fxff)
from torch.fx import GraphModule
from torch.nn.intrinsic import _FusedModule
from torch.quantization.utils import get_combined_dict

from mmrazor.registry import MODELS
from mmrazor.structures.quantization import (CheckArgs, DefalutQconfigs,
                                             QuantizeScheme, SupportQtypes)
from ..task_modules.fx import custom_trace


@MODELS.register_module()
class TensorRTQuantizer:

    def __init__(self, skipped_method=None, qconfig=None) -> None:
        self.skipped_method = skipped_method
        self.qconfig = qconfig

    def prepare(self, model, concreate_args=None) -> GraphModule:
        _swap_ff_with_fxff(model)
        model = custom_trace(
            model,
            concrete_args=concreate_args,
            customed_skipped_method=self.skipped_method)
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, self.qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, self.qconfig)
        return model

    def _insert_fake_quantize_for_act_quant(self, model: GraphModule,
                                            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = '_post_act_fake_quantizer'
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(
            node_to_quantize_output).keys()

        for node in node_to_quantize_output:
            fake_quantizer = qconfig.activation()
            quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            self.logger.info('Insert act quant {}'.format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node('call_module',
                                                  quantizer_name, (node, ), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(
                        _node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model

    def _weight_quant(self, model: GraphModule, qconfig):
        self.logger.info('Replace module to qat module.')
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model

    def _qat_swap_modules(self, root: GraphModule,
                          additional_qat_module_mapping: Dict[Callable,
                                                              Callable]):
        all_mappings = get_combined_dict(get_default_qat_module_mappings(),
                                         additional_qat_module_mapping)
        root = self._convert(root, all_mappings, inplace=True)
        return root

    def _convert(self, module, mapping=None, inplace=False, scope=''):
        if mapping is None:
            mapping = get_default_static_quant_module_mappings()

        if not inplace:
            module = deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            # fused modules are swapped as one unit
            new_scope = '{}.{}'.format(scope, name) if scope != '' else name
            if new_scope in self.exclude_module_name:
                self.logger.info('Skip quant layer: ' + new_scope)
                continue
            if not isinstance(mod, _FusedModule):
                self._convert(mod, mapping, True, new_scope)
            reassign[name] = swap_module(mod, mapping, {})
            if isinstance(mod, torch.nn.ConvTranspose2d):
                if hasattr(
                        reassign[name], 'weight_fake_quant'
                ) and reassign[name].weight_fake_quant.ch_axis != -1:
                    reassign[name].weight_fake_quant.ch_axis = 1
                    reassign[
                        name].weight_fake_quant.activation_post_process.ch_axis = 1  # noqa: E501
        for key, value in reassign.items():
            module._modules[key] = value

        return module

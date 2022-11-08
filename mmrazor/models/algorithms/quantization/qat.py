# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.structures import BaseDataElement
from torch.fx import GraphModule

from mmrazor.registry import MODELS
from ...fx_models import FXModelWrapper
from ...task_modules.fx.tracer import custom_trace
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class QuantizationAwareTraining(BaseAlgorithm):

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 init_cfg=None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        super().__init__(architecture, data_preprocessor, init_cfg)
        self.quantizer = MODELS.build(quantizer)
        self.observers_enabled = True
        self.fake_quants_enabled = True
        self.graph = self._prepare(self.architecture)

    def _prepare(self, model: nn.Module) -> GraphModule:
        graph = self.quantizer.prepare(model)
        return graph

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:

        assert mode in [
            'tensor', 'loss', 'tensor'
        ], f'Invalid mode "{mode}". Only supports loss, predict and tensor mode'
        self.graph.to_mode(mode)
        return self.graph(inputs, data_samples)

    def calib_step(self, data):
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='tensor')

    def convert(self):
        self.architecture = self.quantizer.convert(self.architecture)

    @property
    def state(self):
        return (self.observers_enabled, self.fake_quants_enabled)

    @state.setter
    def state(self, state):
        observers_enabled, fake_quants_enabled = state
        for name, submodule in self.architecture.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantize):
                if observers_enabled:
                    submodule.enable_observer()
                else:
                    submodule.disable_observer()

                if fake_quants_enabled:
                    submodule.enable_fake_quant()
                else:
                    submodule.disable_fake_quant()

        self.observers_enabled = observers_enabled
        self.fake_quants_enabled = fake_quants_enabled

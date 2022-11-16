# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Set, Union

import torch
import torch.fx
from torch.ao.quantization.fx.prepare import is_activation_post_process_node
from torch.fx import Graph
from torch.fx.graph_module import GraphModule, _copy_attr


class MMGraphModule(GraphModule):
    _graph_map: Dict[str, Graph] = dict()
    _mode = None

    def __init__(self, root, graph, class_name: str = 'MMGraphModule') -> None:
        if isinstance(graph, Graph):
            super().__init__(root, graph, class_name)

        elif isinstance(graph, (dict)):
            assert len(graph) > 0, '`graph` should has 1 Graph at least.'

            self._custom_init(root, graph, class_name)

    def _custom_init(self, root, graphs, class_name):
        self._graph_map = graphs

        super(GraphModule, self).__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, torch.nn.Module):
            if hasattr(root, 'training'):
                self.training = root.training
            for _, graph in graphs.items():
                for node in graph.nodes:
                    if node.op in ['get_attr', 'call_module']:
                        assert isinstance(node.target, str)
                        _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            raise NotImplementedError(
                'Customed GraphModule do not support `root` of `dict` type')
        else:
            raise RuntimeError('Unsupported type ' + str(root) +
                               ' passed for root!')

        # Set the first graph as default graph.
        self.graph = list(graphs.values())[0]
        self.mode = list(graphs.keys())[0]

        # Store the Tracer class responsible for creating a Graph separately
        # as part of the GraphModule state, except when the Tracer is defined
        # in a local namespace. Locally defined Tracers are not pickleable.
        # This is needed because torch.package will serialize a GraphModule
        # without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls = None
        if self.graph._tracer_cls and (
                '<locals>' not in self.graph._tracer_cls.__qualname__):
            self._tracer_cls = self.graph._tracer_cls

        self._tracer_extras = {}
        if self.graph._tracer_extras:
            self._tracer_extras = self.graph._tracer_extras

        # Dictionary to store metadata
        self.meta: Dict[str, Any] = {}

    def to_mode(self, mode):
        graph = self._graph_map.get(mode, None)
        if graph:
            self.graph = graph
            self.recompile()
            self._mode = mode
        else:
            raise KeyError(
                f'`self._graph_map` has no graph named `{mode}`, expecting one of: {list(self._graph_map.keys())}'  # noqa: E501
            )

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self.to_mode(mode)

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`
        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.
        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            self.to_mode(mode)
            results = self(**data)
        elif isinstance(data, (list, tuple)):
            self.to_mode(mode)
            results = self(*data)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results


class MMFusedGraphModule(MMGraphModule):

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]],
                 graph: Dict[str, Graph], preserved_attr_names: Set[str]):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return MMFusedGraphModule(fake_mod, copy.deepcopy(self.graph),
                                  copy.deepcopy(self.preserved_attr_names))


class MMObservedGraphModule(MMGraphModule):

    def __init__(self, root: Union[torch.nn.Module, Dict[str, Any]],
                 graph: Dict[str, Graph], preserved_attr_names: Set[str]):
        self.preserved_attr_names = set([
            '_activation_post_process_map', '_activation_post_process_indexes',
            '_patterns', '_qconfig_map', '_prepare_custom_config_dict',
            '_equalization_qconfig_map', '_node_name_to_scope',
            '_qconfig_dict', '_is_qat', '_observed_node_names'
        ]).union(preserved_attr_names)
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return MMObservedGraphModule(fake_mod, copy.deepcopy(self.graph),
                                     copy.deepcopy(self.preserved_attr_names))

    def sync_observer_insertion(self):
        """Sync current graph observer insertion with other modes, especially
        `loss` mode to `predict` mode."""
        for mode in list(self._graph_map.keys()):
            if mode == self.mode:
                continue
            self._sync_observer_between_two_graphs(self.graph,
                                                   self._graph_map[mode])

    def _sync_observer_between_two_graphs(self, graphA, graphB):
        nodeA = graphA._root
        nodeB = graphB._root
        modules = dict(self.named_modules(remove_duplicate=False))
        while nodeA.name != 'output':
            if is_activation_post_process_node(nodeA, modules):
                nodeB = nodeB._prev
                with graphB.inserting_after(nodeB):
                    nodeB = graphB.create_node('call_module', nodeA.name,
                                               (nodeB, ), {})

            nodeA = nodeA._next
            nodeB = nodeB._next

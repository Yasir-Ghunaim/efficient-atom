import torch
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar, override
from typing import cast
from contextlib import ExitStack


from jmp.models.gemnet.backbone_extract_features import GemNetOCWithFeatureExtraction
from jmp.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone

from jmp.models.gemnet.backbone import GOCBackboneOutput
from jmp.tasks.finetune.base import FinetuneModelBase, FinetuneConfigBase
from jmp.tasks.finetune.energy_forces_base import EnergyForcesModelBase

from jmp.tasks.finetune.rmd17 import RMD17Config, RMD17Model
from jmp.tasks.finetune.qm9 import QM9Config, QM9Model
from jmp.tasks.finetune.md22 import MD22Config, MD22Model
from jmp.tasks.finetune.spice import SPICEConfig, SPICEModel
from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel
from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel

TConfig = TypeVar("TConfig", bound=FinetuneConfigBase)

class FinetuneModelBaseFeatureExtraction(FinetuneModelBase[TConfig]):
    @override
    def _construct_backbone(self):
        if self.config.model_name == "gemnet":
            backbone = GemNetOCWithFeatureExtraction(self.config.backbone, **dict(self.config.backbone))
        elif self.config.model_name == "equiformer_v2":
            # Use default hyperparameters for feature extraction
            self.config.backbone.max_radius = 12.0
            self.config.backbone.max_neighbors = 20
            backbone = EquiformerV2Backbone(**dict(self.config.backbone))
        return backbone
    
    @override
    def forward(self, data: BaseData):
        if self.config.model_name == "gemnet":
            atomic_numbers = data.atomic_numbers - 1
            h = self.embedding(atomic_numbers)  # (N, d_model)
            out = cast(GOCBackboneOutput, self.backbone(data, h=h))
            features_dict = out["features_dict"]
        elif self.config.model_name == "equiformer_v2":
            features_dict = {}
            out = self.backbone(data)
            if out is None: #in case the graph includes atoms with high atomic number (>90)
                return None
            node_embedding = out["node_embedding"].embedding
            node_size = node_embedding.shape[0]
            features_dict['node'] = node_embedding.reshape(node_size, -1)

        return features_dict


class EnergyForcesModelBaseFeatureExtraction(EnergyForcesModelBase[TConfig], FinetuneModelBaseFeatureExtraction[TConfig]):

    @override
    def forward(self, data: BaseData):
        with ExitStack() as stack:
            if self.config.gradient_forces or (
                self.config.pretrain_output_head.enabled
                and self.config.pretrain_output_head.gradient_forces
            ):
                stack.enter_context(torch.inference_mode(mode=False))
                stack.enter_context(torch.enable_grad())

                data.pos.requires_grad_(True)
                data = self.generate_graphs_transform(data)

            if self.config.model_name == "gemnet":
                atomic_numbers = data.atomic_numbers - 1
                h = self.embedding(atomic_numbers)
                out: GOCBackboneOutput = self.backbone(data, h=h)
                features_dict = out["features_dict"]

            elif self.config.model_name == "equiformer_v2":
                features_dict = {}
                out = self.backbone(data)
                node_embedding = out["node_embedding"].embedding
                node_size = node_embedding.shape[0]
                features_dict['node'] = node_embedding.reshape(node_size, -1)

        return features_dict


class RMD17ModelFeatureExtraction(RMD17Model, EnergyForcesModelBaseFeatureExtraction[RMD17Config]):
    """
    This class inherits from RMD17Model and EnergyForcesModelBaseFeatureExtraction.
    """

class QM9ModelFeatureExtraction(QM9Model, FinetuneModelBaseFeatureExtraction[QM9Config]):
    """
    This class inherits from QM9Model and FinetuneModelBaseFeatureExtraction.
    """

class MD22ModelFeatureExtraction(MD22Model, EnergyForcesModelBaseFeatureExtraction[MD22Config]):
    """
    This class inherits from MD22Model and EnergyForcesModelBaseFeatureExtraction.
    """

class SPICEModelFeatureExtraction(SPICEModel, EnergyForcesModelBaseFeatureExtraction[SPICEConfig]):
    """
    This class inherits from SPICEModel and EnergyForcesModelBaseFeatureExtraction.
    """

class MatbenchModelFeatureExtraction(MatbenchModel, FinetuneModelBaseFeatureExtraction[MatbenchConfig]):
    """
    This class inherits from MatbenchModel and FinetuneModelBaseFeatureExtraction.
    """

class QMOFModelFeatureExtraction(QMOFModel, FinetuneModelBaseFeatureExtraction[QMOFConfig]):
    """
    This class inherits from QMOFModel and FinetuneModelBaseFeatureExtraction.
    """
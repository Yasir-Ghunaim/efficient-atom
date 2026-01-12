import torch
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from einops import pack

from typing_extensions import TypeVar, override

from jmp.tasks.pretrain import PretrainModel
from jmp.tasks.pretrain.module import Output, PretrainConfig
from jmp.models.gemnet.backbone import GOCBackboneOutput
from jmp.models.gemnet.backbone_extract_features import GemNetOCWithFeatureExtraction
from jmp.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone


TConfig = TypeVar(
    "TConfig", bound=PretrainConfig, default=PretrainConfig, infer_variance=True
)


# class OutputWithFeatureExtraction(Output):
#     @override
#     def forward(self, data: BaseData, backbone_out: GOCBackboneOutput):
#         energy = backbone_out["energy"]
#         forces = backbone_out["forces"]
#         features_dict = backbone_out["features_dict"]
#         V_st = backbone_out["V_st"]
#         idx_t = backbone_out["idx_t"]

#         batch: torch.Tensor = data.batch
#         n_molecules = int(torch.max(batch).item() + 1)
#         n_atoms = data.atomic_numbers.shape[0]

#         energy_list: list[torch.Tensor] = []
#         forces_list: list[torch.Tensor] = []

#         for energy_mlp, forces_mlp, task in zip(
#             self.out_energy, self.out_forces, self.config.tasks
#         ):
#             E_t = energy_mlp(energy)  # (n_atoms, 1)
#             E_t = scatter(
#                 E_t,
#                 batch,
#                 dim=0,
#                 dim_size=n_molecules,
#                 reduce=task.node_energy_reduction,
#             )
#             energy_list.append(E_t)  # (bsz, 1)

#             F_st = forces_mlp(forces)  # (n_edges, 1)
#             F_st = F_st * V_st  # (n_edges, 3)
#             F_t = scatter(F_st, idx_t, dim=0, dim_size=n_atoms, reduce="sum")
#             forces_list.append(F_t)  # (n_atoms, 3)

#         E, _ = pack(energy_list, "bsz *")
#         F, _ = pack(forces_list, "n_atoms p *")

#         return E, F, features_dict

class PretrainModelWithFeatureExtraction(PretrainModel[TConfig]):
    @override
    def _construct_backbone(self):
        if self.config.model_name == "gemnet":
            backbone = GemNetOCWithFeatureExtraction(self.config.backbone, **dict(self.config.backbone))
        elif self.config.model_name == "equiformer_v2":
            backbone = EquiformerV2Backbone(**dict(self.config.backbone))
        return backbone

    @override
    def __init__(self, hparams: TConfig):
        self._model_validate_config(hparams)
        super().__init__(hparams)
        # self.output = OutputWithFeatureExtraction(self.config)

    
    @override
    def forward(self, batch: BaseData):
        if self.config.model_name == "gemnet":
            h = self.embedding(batch)
            out: GOCBackboneOutput = self.backbone(batch, h=h)
            features_dict = out["features_dict"]
        elif self.config.model_name == "equiformer_v2":
            features_dict = {}
            out = self.backbone(batch)
            node_embedding = out["node_embedding"].embedding
            node_size = node_embedding.shape[0]
            features_dict['node'] = node_embedding.reshape(node_size, -1)
        return features_dict 
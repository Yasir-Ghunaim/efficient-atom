import torch
from torch import nn
from torch_geometric.data.data import BaseData

from jmp.models.gemnet.backbone import GemNetOCBackbone

class GemNetOCWithFeatureExtraction(GemNetOCBackbone):
    def forward(
        self,
        data: BaseData,
        *,
        h: torch.Tensor,
    ):
        pos = data.pos
        # batch = data.batch
        # atomic_numbers = data.atomic_numbers.long()
        num_atoms = data.atomic_numbers.shape[0]

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        idx_s, idx_t = main_graph["edge_index"]

        bases: BasesOutput = self.bases(
            data,
            h=h,
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )
        m = bases.m

        # Embedding block
        # h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        # m = self.edge_emb(h, bases.rbf_main, main_graph["edge_index"])
        # (nEdges_main, emb_size_edge)

        features_dict = {}
        features_dict["idx_t"] = idx_t

        x_E, x_F = self.out_blocks[0](h, m, bases.output, idx_t, data=data)
        # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
        xs_E, xs_F = [x_E], [x_F]

        for i in range(self.num_blocks):
            if self.config.unique_basis_per_layer:
                bases: BasesOutput = self.per_layer_bases[i](
                    data,
                    h=h,
                    main_graph=main_graph,
                    a2a_graph=a2a_graph,
                    a2ee2a_graph=a2ee2a_graph,
                    qint_graph=qint_graph,
                    trip_idx_e2e=trip_idx_e2e,
                    trip_idx_a2e=trip_idx_a2e,
                    trip_idx_e2a=trip_idx_e2a,
                    quad_idx=quad_idx,
                    num_atoms=num_atoms,
                )
                m = m + bases.m

            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases.qint,
                bases_e2e=bases.e2e,
                bases_a2e=bases.a2e,
                bases_e2a=bases.e2a,
                basis_a2a_rad=bases.a2a_rad,
                basis_atom_update=bases.atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_E, x_F = self.out_blocks[i + 1](h, m, bases.output, idx_t, data=data)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_E.append(x_E)
            xs_F.append(x_F)

        # Global output block for final predictions
        if self.regress_forces:
            assert self.direct_forces, "Only direct forces are supported for now."
            x_F = self.out_mlp_F(
                torch.cat(xs_F, dim=-1), data=data, edge_index=main_graph["edge_index"]
            )
        else:
            x_F = None

        if self.regress_energy:
            x_E = self.out_mlp_E(
                torch.cat(xs_E, dim=-1), data=data, edge_index=main_graph["edge_index"]
            )
        else:
            x_E = None

        features_dict[f'node'] = x_E
        features_dict[f'edge'] = x_F

        out: GOCBackboneOutput = {
            # "energy": x_E,
            # "forces": x_F,
            "features_dict": features_dict,
            # "V_st": main_graph["vector"],
            # "D_st": main_graph["distance"],
            # "idx_s": idx_s,
            # "idx_t": idx_t,
        }
        return out

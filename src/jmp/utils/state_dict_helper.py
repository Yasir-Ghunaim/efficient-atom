import re

def update_gemnet_state_dict_keys(full_state_dict, is_pretrain=False):
    """
    Updates the keys in a state dictionary by adding 'bases' after 'backbone' 
    for specific patterns.

    Args:
        full_state_dict (dict): Original state dictionary with keys to be updated.

    Returns:
        dict: Updated state dictionary with modified keys.
    """

    keys_to_update = [
        "radial_basis.rbf.offset",
        "radial_basis.rbf.temps",
        "cbf_basis_qint.radial_basis.rbf.offset",
        "cbf_basis_qint.radial_basis.rbf.temps",
        "sbf_basis_qint.radial_basis.rbf.offset",
        "sbf_basis_qint.radial_basis.rbf.temps",
        "radial_basis_aeaint.rbf.offset",
        "radial_basis_aeaint.rbf.temps",
        "cbf_basis_aeint.radial_basis.rbf.offset",
        "cbf_basis_aeint.radial_basis.rbf.temps",
        "cbf_basis_eaint.radial_basis.rbf.offset",
        "cbf_basis_eaint.radial_basis.rbf.temps",
        "radial_basis_aint.rbf.offset",
        "radial_basis_aint.rbf.temps",
        "cbf_basis_tint.radial_basis.rbf.offset",
        "cbf_basis_tint.radial_basis.rbf.temps",
        "mlp_rbf_qint.linear.weight",
        "mlp_cbf_qint.weight",
        "mlp_sbf_qint.weight",
        "mlp_rbf_aeint.linear.weight",
        "mlp_cbf_aeint.weight",
        "mlp_rbf_eaint.linear.weight",
        "mlp_cbf_eaint.weight",
        "mlp_rbf_aint.weight",
        "mlp_rbf_tint.linear.weight",
        "mlp_cbf_tint.weight",
        "mlp_rbf_h.linear.weight",
        "mlp_rbf_out.linear.weight",
        "edge_emb.dense.linear.weight"
    ]

    # Remove the keys for the head
    keys_to_remove = [
        "module.module.out_energy.linear.weight",
        "module.module.out_forces.linear.weight"
    ]   

    # For fine-tuning only, remove keys starting with "backbone.out_mlp_F" or matching the pattern
    if not is_pretrain:
        keys_to_remove += [
            key for key in full_state_dict.keys() 
            if key.startswith("module.module.out_mlp_F") or 
            re.match(r"module\.module\.out_blocks\.\d+\.seq_forces", key) or
            re.match(r"module\.module\.out_blocks\.\d+\.dense_rbf_F", key)
        ]
    
    
    # Create a new state dictionary without the unwanted keys
    full_state_dict = {
        key: value for key, value in full_state_dict.items()
        if key not in keys_to_remove
    }

    # Fix key for embeddings
    if is_pretrain:
        full_state_dict = {
            key.replace("module.module.atom_emb.embeddings.weight", "embedding.atom_embedding.weight"): value
            for key, value in full_state_dict.items()
        }
    else:
        full_state_dict = {
            key.replace("module.module.atom_emb.embeddings.weight", "embedding.weight"): value
            for key, value in full_state_dict.items()
        }
    # Fix key for output blocks
    full_state_dict = {
        re.sub(r"^module\.module\.out_mlp_E\.(.*)", r"backbone.out_mlp_E.out_mlp.\1", key): value
        for key, value in full_state_dict.items()
    }
    full_state_dict = {
        re.sub(r"^module\.module\.out_mlp_F\.(.*)", r"backbone.out_mlp_F.out_mlp.\1", key): value
        for key, value in full_state_dict.items()
    }

    # Fix key for bases
    updated_state_dict = {}
    for key, value in full_state_dict.items():
        # Check if the key matches one of the keys to update
        for keyword in keys_to_update:
            if key.startswith(f"module.module.{keyword}"):
                # Update the key to include "bases" after "backbone"
                new_key = key.replace("module.module.", "backbone.bases.", 1)
                updated_state_dict[new_key] = value
                break
        else:
            # Keep the key unchanged if it doesn't match
            updated_state_dict[key] = value

    return updated_state_dict
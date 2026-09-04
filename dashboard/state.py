def sync_preset_state(state, preset_key, presets):
    """Load a preset into both the model state and every Streamlit widget."""
    preset = presets[preset_key]
    features = preset["features"].copy()

    state["txn_features"] = features
    state["active_preset_name"] = preset["name"]
    state["txn_id"] = f"TXN-PRESET-{preset_key.upper()}"
    state["input_transaction_id"] = state["txn_id"]
    for feature_name, value in features.items():
        state[f"input_{feature_name}"] = value
    state["auto_trigger"] = True

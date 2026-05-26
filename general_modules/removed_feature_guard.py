"""Fail-fast checks for legacy branch artifacts removed from this checkout."""

REMOVED_MODES = {
    "train_prior",
    "train_with_prior",
}

REMOVED_MODEL_NAMES = {
    "meshgraphnets-v",
    "meshgraphnets_vae",
    "vae-mgn",
}

REMOVED_CONFIG_KEYS = {
    "use_vae",
    "vae_latent_dim",
    "vae_mp_layers",
    "vae_graph_aware",
    "free_bits",
    "posterior_min_std",
    "lambda_mmd",
    "lambda_kl",
    "lambda_det",
    "alpha_recon",
    "beta_aux",
    "num_vae_samples",
    "vae_valid_prior_samples",
    "fit_latent_gmm",
    "gmm_components",
    "gmm_covariance_type",
    "gmm_reg_covar",
    "train_conditional_prior",
    "use_conditional_prior",
    "prior_temperature",
    "prior_mixture_components",
    "prior_hidden_dim",
    "prior_mp_layers",
    "prior_min_std",
    "prior_loss_type",
    "prior_epochs",
    "prior_learningr",
    "prior_batch_size",
    "prior_num_workers",
    "prior_val_interval",
    "prior_diagnose_interval",
    "prior_mc_samples",
    "resume_prior",
    "num_z",
}

REMOVED_CHECKPOINT_KEYS = {
    "gmm_params",
    "conditional_prior_state_dict",
    "conditional_prior_config",
    "conditional_prior_metrics",
    "valid_prior_loss",
    "valid_prior_samples",
}

REMOVED_STATE_PREFIXES = (
    "model.vae_encoder.",
    "model.aux_decoder.",
    "model.z_fusers.",
    "model.ms_z_fusers_pre.",
    "model.ms_z_fusers_post.",
    "model.ms_z_fusers_coarsest.",
)


def _format_list(values):
    return ", ".join(sorted(str(v) for v in values))


def validate_config(config, source="configuration"):
    """Reject old branch modes, model names, and keys."""
    mode = str(config.get("mode", "")).lower()
    if mode in REMOVED_MODES:
        raise ValueError(f"Unsupported mode '{mode}' in {source}; this checkout supports only 'train' and 'inference'.")

    model_name = str(config.get("model", "")).lower()
    if model_name in REMOVED_MODEL_NAMES:
        raise ValueError(f"{source} uses removed model '{model_name}'. Use 'MeshGraphNets'.")

    removed_keys = REMOVED_CONFIG_KEYS.intersection(config.keys())
    if removed_keys:
        raise ValueError(f"{source} contains removed legacy branch keys: {_format_list(removed_keys)}")


def validate_checkpoint(checkpoint, source="checkpoint"):
    """Reject checkpoints saved from the removed legacy branch."""
    top_level = REMOVED_CHECKPOINT_KEYS.intersection(checkpoint.keys())
    if top_level:
        raise ValueError(f"{source} contains removed legacy branch artifacts: {_format_list(top_level)}")

    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, dict):
        validate_config(model_config, f"{source} model_config")

    for state_key in ("model_state_dict", "ema_state_dict"):
        state_dict = checkpoint.get(state_key)
        if not isinstance(state_dict, dict):
            continue
        bad = [
            key for key in state_dict
            if any(key.startswith(prefix) or key.startswith(f"module.{prefix}") for prefix in REMOVED_STATE_PREFIXES)
        ]
        if bad:
            sample = ", ".join(sorted(bad)[:5])
            raise ValueError(f"{source} {state_key} contains removed legacy branch weights: {sample}")

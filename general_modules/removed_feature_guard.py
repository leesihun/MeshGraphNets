"""Guards for config/checkpoint artifacts from the removed probabilistic branch."""

REMOVED_CONFIG_KEYS = {
    'use_vae',
    'lambda_mmd',
    'alpha_recon',
    'beta_aux',
    'lambda_det',
    'free_bits',
    'posterior_min_std',
    'num_z',
    'num_vae_samples',
    'fit_latent_gmm',
    'gmm_components',
    'gmm_covariance_type',
    'gmm_reg_covar',
    'train_conditional_prior',
    'resume_prior',
    'vae_valid_prior_samples',
}

REMOVED_CONFIG_PREFIXES = (
    'vae_',
    'gmm_',
    'prior_',
)

REMOVED_CHECKPOINT_KEYS = {
    'conditional_prior_state_dict',
    'conditional_prior_config',
    'conditional_prior_metrics',
    'gmm_params',
    'valid_prior_loss',
    'valid_prior_samples',
}

REMOVED_STATE_PREFIXES = (
    'model.vae_encoder.',
    'model.aux_decoder.',
    'model.z_fusers.',
    'model.ms_z_fusers_',
    'vae_encoder.',
    'aux_decoder.',
    'z_fusers.',
    'ms_z_fusers_',
)


def _removed_config_keys(config):
    removed = []
    for key in config:
        key_l = str(key).lower()
        if key_l in REMOVED_CONFIG_KEYS:
            removed.append(key)
            continue
        if any(key_l.startswith(prefix) for prefix in REMOVED_CONFIG_PREFIXES):
            removed.append(key)
    return sorted(set(removed), key=str)


def validate_no_removed_config(config, source='configuration'):
    """Raise if a config still requests the removed probabilistic branch."""
    model = str(config.get('model', '')).lower()
    if model in {'meshgraphnets-v', 'meshgraphnets_v', 'hi-mgn-v', 'hi_mgn_v'}:
        raise ValueError(
            f"{source} uses removed model '{config.get('model')}'. "
            "Use 'MeshGraphNets' and remove old probabilistic keys."
        )

    removed = _removed_config_keys(config)
    if removed:
        keys = ', '.join(str(k) for k in removed)
        raise ValueError(
            f"{source} contains removed probabilistic keys: {keys}. "
            "Remove them before running this deterministic codebase."
        )


def validate_no_removed_checkpoint(checkpoint, source='checkpoint'):
    """Raise if a checkpoint belongs to the removed probabilistic branch."""
    removed = sorted(k for k in REMOVED_CHECKPOINT_KEYS if k in checkpoint)

    model_config = checkpoint.get('model_config')
    if isinstance(model_config, dict):
        removed.extend(f"model_config.{k}" for k in _removed_config_keys(model_config))
        model_name = str(model_config.get('model', '')).lower()
        if model_name in {'meshgraphnets-v', 'meshgraphnets_v', 'hi-mgn-v', 'hi_mgn_v'}:
            removed.append(f"model_config.model={model_config.get('model')}")

    state_dict = checkpoint.get('model_state_dict', {})
    if isinstance(state_dict, dict):
        for key in state_dict:
            if any(str(key).startswith(prefix) for prefix in REMOVED_STATE_PREFIXES):
                removed.append(f"model_state_dict.{key}")
                break

    ema_state_dict = checkpoint.get('ema_state_dict', {})
    if isinstance(ema_state_dict, dict):
        for key in ema_state_dict:
            key_l = str(key)
            stripped = key_l[len('module.'):] if key_l.startswith('module.') else key_l
            if any(stripped.startswith(prefix) for prefix in REMOVED_STATE_PREFIXES):
                removed.append(f"ema_state_dict.{key}")
                break

    if removed:
        keys = ', '.join(sorted(set(removed), key=str))
        raise ValueError(
            f"{source} belongs to the removed probabilistic branch ({keys}). "
            "Retrain or provide a deterministic MeshGraphNets checkpoint."
        )

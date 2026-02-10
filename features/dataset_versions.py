from .transformation import assemble_feature_set

def load_dataset_version(df_raw, version: str):
    """
    Maps version names to pipeline configurations.
    """

    versions = {
        "v0_raw": dict(apply_cleaning=False),

        "v1_clipped": dict(
            apply_cleaning=True,
            impute_bmi=False,
            clip_outliers=True,
            drop_missing=False
        ),

        "v2_dropped": dict(
            apply_cleaning=True,
            impute_bmi=False,
            clip_outliers=True,
            drop_missing=True
        ),

        "v3_imputed": dict(
            apply_cleaning=True,
            impute_bmi=True,
            clip_outliers=True,
            drop_missing=False
        ),
        "v4_dropped_no_impute": dict(
            apply_cleaning=True,
            impute_bmi=False,
            clip_outliers=True,
            drop_missing=True
        ),
    }

    if version not in versions:
        raise ValueError(f"Unknown dataset version '{version}'. Valid versions: {list(versions.keys())}")

    # unpack config into assemble_feature_set(...)
    return assemble_feature_set(df_raw, **versions[version])
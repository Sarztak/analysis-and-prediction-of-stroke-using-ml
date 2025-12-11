from azure.ai.ml import MLClient, Input, dsl, load_component
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# ----------------------------------------------------
# 1. Connect to Azure ML workspace
# ----------------------------------------------------
def get_ml_client() -> MLClient:
    return MLClient.from_config(credential=DefaultAzureCredential())

ml_client = get_ml_client()


# ----------------------------------------------------
# 2. Load your components from YAML
# ----------------------------------------------------

# Preprocessing component
prep_comp = load_component(
    source="components/stroke_data_prep_component.yaml"
)

# Model evaluation component
eval_comp = load_component(
    source="monitoring/evaluate_model.yaml"
)

# Drift detection component
drift_comp = load_component(
    source="monitoring/stroke_drift.yaml"
)


# ----------------------------------------------------
# 3. End-to-End Pipeline Definition
# ----------------------------------------------------
@dsl.pipeline(
    compute="drift-cluster",
    description="End-to-end stroke pipeline: preprocess → eval → drift detection",
)
def stroke_pipeline(version: str = "v2_dropped"):
    """End-to-end stroke pipeline:
    1) Preprocess raw data into train/test/modified
    2) Evaluate model on baseline test data
    3) Evaluate model on modified test data
    4) Run drift detection between baseline & modified data
    """

    # ---- pipeline Input ----
    raw_data = Input(
        type=AssetTypes.URI_FILE,
        path="azureml:stroke-raw-data:2",
    )

    model_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path="azureml:stroke_model_folder:1",
    )

    # ---------- Step 1: Preprocess raw data ----------
    prep_step = prep_comp(
        raw_data=raw_data,
        version=version,
    )
    # train_output, test_output, modified_output

    # ---------- Step 2: Evaluate on baseline test ----------
    eval_base = eval_comp(
        model_dir=model_folder,
        test_data=prep_step.outputs.test_output,
    )

    # ---------- Step 3: Evaluate on modified test ----------
    eval_mod = eval_comp(
        model_dir=model_folder,
        test_data=prep_step.outputs.modified_output,
    )

    # ---------- Step 4: Drift detection ----------
    drift_job = drift_comp(
        baseline_data=prep_step.outputs.test_output,
        modified_data=prep_step.outputs.modified_output,
    )

    # ---------- Pipeline outputs ----------
    return {
        "train_data": prep_step.outputs.train_output,
        "test_data": prep_step.outputs.test_output,
        "modified_data": prep_step.outputs.modified_output,
        "base_metrics": eval_base.outputs.metrics_output,
        "mod_metrics": eval_mod.outputs.metrics_output,
        "drift_report": drift_job.outputs.drift_report,
    }


if __name__ == "__main__":
    job = stroke_pipeline()
    job = ml_client.jobs.create_or_update(job)
    print("Submitted end-to-end stroke pipeline!")
    print(job.studio_url)
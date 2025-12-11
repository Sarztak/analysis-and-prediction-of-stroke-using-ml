from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml import load_component
from azure.identity import DefaultAzureCredential


def get_ml_client() -> MLClient:
    """Create an MLClient from the local Azure ML config.

    This assumes you have a valid config.json in your working directory
    or that the environment is already configured for MLClient.from_config().
    """
    return MLClient.from_config(credential=DefaultAzureCredential())


# Create a global MLClient so components and the pipeline can use it.
ml_client = get_ml_client()


# ------------------------------------------------------------------
# 1. (Optional) Component registration from YAML
# ------------------------------------------------------------------
# Example: register / update components from local YAML definitions.
# Uncomment and adjust the paths if you want to re-register components.
#
# from azure.ai.ml import load_component
#
# eval_comp = load_component(source="monitoring/evaluate_model.yaml")
# eval_comp_registered = ml_client.components.create_or_update(eval_comp)
#
# drift_comp = load_component(source="monitoring/stroke_drift.yaml")
# drift_comp_registered = ml_client.components.create_or_update(drift_comp)
#
# print("Registered eval component:", eval_comp_registered.name, eval_comp_registered.version)
# print("Registered drift component:", drift_comp_registered.name, drift_comp_registered.version)


# ------------------------------------------------------------------
# 2. Get the existing registered components
# ------------------------------------------------------------------
model_eval_comp = ml_client.components.get(
    name="stroke_model_eval",
    label="latest",
)

stroke_drift_comp = ml_client.components.get(
    name="stroke_data_drift",
    version="1",
)


# ------------------------------------------------------------------
# 3. Define the monitoring pipeline
# ------------------------------------------------------------------
@dsl.pipeline(
    default_compute="drift-cluster",
)
def stroke_monitoring_pipeline(
    baseline_data: Input,
    modified_data: Input,
    model_dir: Input,
):
    """Monitoring pipeline for the stroke model.

    Steps:
    1. Evaluate the model on the baseline data.
    2. Evaluate the model on the modified (current) data.
    3. Run data drift detection between baseline and modified data.
    """

    # 1. Evaluation on baseline data
    eval_base = model_eval_comp(
        model_dir=model_dir,
        test_data=baseline_data,
    )

    # 2. Evaluation on modified (current) data
    eval_mod = model_eval_comp(
        model_dir=model_dir,
        test_data=modified_data,
    )

    # 3. Data drift detection
    drift_job = stroke_drift_comp(
        baseline_data=baseline_data,
        modified_data=modified_data,
    )

    # Expose metrics & report as pipeline outputs
    return {
        "base_metrics": eval_base.outputs.metrics_output,
        "mod_metrics": eval_mod.outputs.metrics_output,
        "drift_report": drift_job.outputs.drift_report,
    }


# ------------------------------------------------------------------
# 4. Main entry point
# ------------------------------------------------------------------
def main() -> None:
    """Submit the stroke monitoring pipeline job to Azure ML.

    - Create the pipeline job by binding the inputs
      (baseline data, modified data, and model directory).
    - Submit the job via ml_client.jobs.create_or_update().
    - Stream the logs until completion.
    """

    # Bind registered datasets and model folder.
    pipeline_job = stroke_monitoring_pipeline(
        baseline_data=Input(
            type="uri_file",
            path="azureml:stroke-test-final-data:1",
        ),
        modified_data=Input(
            type="uri_file",
            path="azureml:stroke-test-final-data-modified:1",
        ),
        model_dir=Input(
            type="uri_folder",
            path="azureml:stroke_model_folder:1",
        ),
    )

    pipeline_job.settings.force_rerun = True

    # Submit the job to Azure ML
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="stroke_monitoring",
    )
    
    # Stream the logs until the job finishes
    ml_client.jobs.stream(returned_job.name)


if __name__ == "__main__":
    main()

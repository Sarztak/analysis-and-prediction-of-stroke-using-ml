from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

def get_ml_client() -> MLClient:
    return MLClient.from_config(credential=DefaultAzureCredential())

ml_client = get_ml_client()

# Load + register component
data_prep_comp = load_component(source="components/stroke_data_prep_component.yaml")
registered = ml_client.components.create_or_update(data_prep_comp)

print("Registered component:", registered.name, registered.version)

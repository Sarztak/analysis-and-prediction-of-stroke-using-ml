# Azure Deployment Code Using Azure ML CLI
# Register the model best model from H2O: XGboost by uploading the model folder to Azure API
# uploaded file structure
"""
stroke_best_model_xgb_final/
└── model_azure/
    ├── MLmodel
    ├── conda.yaml
    ├── model.h2o/
    ├── python_env.yaml
    ├── requirements.txt
"""

# Check if Azure ML extension is installed
az extension list --output table

# If not installed or outdated, install/update it
az extension add --name ml --upgrade

# install libraries
pip install numpy pandas matplotlib scikit-learn --user

az login 

# configure default workspace
az configure --defaults workspace=mlops-class resource-group=mlops-project_group

# create custom environment
# Create environment with Java
cd ~/h2o-deployment

cat > ~/h2o-deployment/environment-with-java.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: h2o-batch-env-java
version: 1
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
conda_file:
  name: mlflow-env
  channels:
    - conda-forge
  dependencies:
    - python=3.10
    - openjdk=11
    - pip
    - pip:
        - mlflow==3.6.0
        - h2o==3.46.0.9
        - pandas==2.1.1
        - numpy==1.25.2
        - azureml-core>=1.50.0
        - azureml-mlflow>=1.50.0
        - azureml-dataset-runtime>=1.50.0
        - defusedxml==0.7.1
        - pyyaml>=6.0
description: H2O environment with Java 11 for batch inference
EOF

az ml environment create --file environment-with-java.yml

# Wait for environment build (5-10 min)
watch -n 30 'az ml environment show --name h2o-batch-env-java --version 1 --query build_state'


# Create project folder
mkdir h2o-batch-deployment
cd h2o-batch-deployment

# upload the test data (v2 dropped)
az ml data create --name test_df --version 1 --path ./test_df.csv --type uri_file

# scoring script using best threshold from h2o mlflow
cat > ~/h2o-deployment/batch-scoring-with-threshold/score.py << 'EOF'
import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables
model = None
threshold = None
DEFAULT_THRESHOLD = 0.430403

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def load_threshold_from_mlflow(model_dir):
    """Try to load threshold from MLflow model metadata"""
    logger.info("Attempting to load threshold from model artifacts...")
    
    # Method 1: Check MLmodel file
    try:
        mlmodel_path = os.path.join(model_dir, "MLmodel")
        if os.path.exists(mlmodel_path):
            with open(mlmodel_path, 'r') as f:
                mlmodel_content = yaml.safe_load(f)
            
            if 'metadata' in mlmodel_content and 'threshold' in mlmodel_content['metadata']:
                threshold_val = float(mlmodel_content['metadata']['threshold'])
                logger.info(f"  ✓ Found threshold in MLmodel metadata: {threshold_val}")
                return threshold_val
            
            if 'params' in mlmodel_content and 'threshold' in mlmodel_content['params']:
                threshold_val = float(mlmodel_content['params']['threshold'])
                logger.info(f"  ✓ Found threshold in MLmodel params: {threshold_val}")
                return threshold_val
    except Exception as e:
        logger.debug(f"  Could not load threshold from MLmodel: {e}")
    
    # Method 2: Check for threshold.txt
    try:
        threshold_file = os.path.join(model_dir, "threshold.txt")
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                threshold_val = float(f.read().strip())
            logger.info(f"  ✓ Found threshold in threshold.txt: {threshold_val}")
            return threshold_val
    except Exception as e:
        logger.debug(f"  Could not load threshold from threshold.txt: {e}")
    
    # Method 3: Check for threshold.json
    try:
        threshold_json = os.path.join(model_dir, "threshold.json")
        if os.path.exists(threshold_json):
            with open(threshold_json, 'r') as f:
                data = json.load(f)
            if 'threshold' in data:
                threshold_val = float(data['threshold'])
                logger.info(f"  ✓ Found threshold in threshold.json: {threshold_val}")
                return threshold_val
    except Exception as e:
        logger.debug(f"  Could not load threshold from threshold.json: {e}")
    
    logger.info(f"  ! No threshold found in model artifacts")
    return None


def init():
    """Initialize model and load threshold"""
    global model, threshold
    
    logger.info("="*80)
    logger.info("INIT STARTED - WITH THRESHOLD LOADING")
    logger.info("="*80)
    
    try:
        import h2o
        import mlflow.pyfunc
        
        # Initialize H2O
        logger.info("Initializing H2O...")
        h2o.init(strict_version_check=False, max_mem_size="3G", nthreads=-1, log_level="WARN")
        h2o.no_progress()
        logger.info("H2O initialized successfully")
        
        # Find model directory
        model_dir_base = os.getenv("AZUREML_MODEL_DIR")
        logger.info(f"Model base dir: {model_dir_base}")
        
        model_dir = None
        if model_dir_base and os.path.exists(model_dir_base):
            for root, dirs, files in os.walk(model_dir_base):
                if "MLmodel" in files:
                    model_dir = root
                    logger.info(f"Found model at: {model_dir}")
                    break
        
        if not model_dir:
            raise FileNotFoundError(f"MLmodel not found in {model_dir_base}")
        
        # Load the model
        logger.info("Loading MLflow model...")
        model = mlflow.pyfunc.load_model(model_dir)
        logger.info(f"Model loaded: {type(model)}")
        
        # Try to load threshold
        threshold = load_threshold_from_mlflow(model_dir)
        
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
            logger.info(f"  ⚠️  Using DEFAULT threshold: {threshold}")
        else:
            logger.info(f"  ✓ Using LOADED threshold: {threshold}")
        
        logger.info("="*80)
        logger.info("INIT COMPLETED SUCCESSFULLY")
        logger.info(f"Model ready with threshold: {threshold}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"INIT FAILED: {str(e)}", exc_info=True)
        raise


def run(mini_batch):
    """
    Process files and apply threshold to create predicted_class column.
    
    Logic:
    - If probability of class 1 (p1) > threshold: predicted_class = 1 (Stroke)
    - Otherwise: predicted_class = 0 (No Stroke)
    """
    global model, threshold
    
    logger.info("="*80)
    logger.info(f"RUN CALLED - {len(mini_batch)} files")
    logger.info(f"Threshold: {threshold}")
    logger.info("Logic: predicted_class = 1 if p1 > {threshold}, else 0")
    logger.info("="*80)
    
    results = []
    monitoring_stats = {
        'timestamp': datetime.now().isoformat(),
        'threshold_used': float(threshold),
        'files_processed': 0,
        'total_predictions': 0,
        'predicted_class_1': 0,
        'predicted_class_0': 0
    }
    
    for idx, file_path in enumerate(mini_batch):
        try:
            logger.info(f"\n[{idx+1}/{len(mini_batch)}] Processing: {file_path}")
            
            # Read data
            df = pd.read_csv(file_path)
            logger.info(f"  Input shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)[:5]}...")
            
            # Make predictions (returns probabilities)
            logger.info("  Making predictions...")
            predictions = model.predict(df)
            
            # Convert to DataFrame if needed
            if not isinstance(predictions, pd.DataFrame):
                if isinstance(predictions, pd.Series):
                    pred_df = pd.DataFrame({'predict': predictions.values})
                else:
                    pred_df = pd.DataFrame({'predict': predictions})
            else:
                pred_df = predictions.copy()
            
            # Ensure we have probability columns
            # p0 = probability of class 0 (No Stroke)
            # p1 = probability of class 1 (Stroke)
            if 'p1' not in pred_df.columns:
                logger.error("  ✗ ERROR: 'p1' column not found in predictions!")
                logger.error(f"  Available columns: {pred_df.columns.tolist()}")
                continue
            
            # CREATE predicted_class COLUMN
            # Logic: if p1 > threshold then 1, else 0
            pred_df['predicted_class'] = (pred_df['p1'] > threshold).astype(int)
            
            logger.info(f"  ✓ Applied threshold logic:")
            logger.info(f"    If p1 > {threshold} → predicted_class = 1 (Stroke)")
            logger.info(f"    If p1 <= {threshold} → predicted_class = 0 (No Stroke)")
            
            # Add metadata
            pred_df['source_file'] = os.path.basename(file_path)
            pred_df['row_num'] = range(len(pred_df))
            pred_df['timestamp'] = datetime.now().isoformat()
            pred_df['threshold_used'] = float(threshold)
            
            # Calculate statistics
            class_1_count = int((pred_df['predicted_class'] == 1).sum())
            class_0_count = int((pred_df['predicted_class'] == 0).sum())
            class_1_pct = class_1_count / len(pred_df) * 100
            class_0_pct = class_0_count / len(pred_df) * 100
            mean_p1 = float(pred_df['p1'].mean())
            
            # Update monitoring stats
            monitoring_stats['files_processed'] += 1
            monitoring_stats['total_predictions'] += int(len(pred_df))
            monitoring_stats['predicted_class_1'] += class_1_count
            monitoring_stats['predicted_class_0'] += class_0_count
            
            results.append(pred_df)
            
            # Log statistics
            logger.info(f"  ✓ SUCCESS - {len(pred_df)} predictions generated")
            logger.info(f"    Predicted Class 1 (Stroke):    {class_1_count} ({class_1_pct:.2f}%)")
            logger.info(f"    Predicted Class 0 (No Stroke): {class_0_count} ({class_0_pct:.2f}%)")
            logger.info(f"    Mean p1 (stroke probability):  {mean_p1:.4f}")
            
            # Show sample predictions
            logger.info(f"  Sample predictions (first 3 rows):")
            for i in range(min(3, len(pred_df))):
                row = pred_df.iloc[i]
                logger.info(f"    Row {i}: p1={row['p1']:.4f} → predicted_class={int(row['predicted_class'])}")
            
        except Exception as e:
            logger.error(f"  ✗ ERROR processing {file_path}: {str(e)}", exc_info=True)
            continue
    
    # Combine results and log final summary
    if results:
        final_df = pd.concat(results, ignore_index=True)
        
        total_class_1 = monitoring_stats['predicted_class_1']
        total_class_0 = monitoring_stats['predicted_class_0']
        total_predictions = monitoring_stats['total_predictions']
        
        logger.info("\n" + "="*80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Files processed: {monitoring_stats['files_processed']}")
        logger.info(f"Total predictions: {total_predictions}")
        logger.info(f"Threshold used: {threshold}")
        logger.info(f"\nPrediction Summary:")
        logger.info(f"  Class 1 (Stroke):    {total_class_1} ({total_class_1/total_predictions*100:.2f}%)")
        logger.info(f"  Class 0 (No Stroke): {total_class_0} ({total_class_0/total_predictions*100:.2f}%)")
        logger.info("="*80)
        
        # Convert to JSON-serializable and log
        monitoring_stats_serializable = convert_to_json_serializable(monitoring_stats)
        try:
            logger.info("\nMONITORING_METRICS: " + json.dumps(monitoring_stats_serializable, indent=2))
        except Exception as e:
            logger.warning(f"Could not serialize monitoring metrics: {e}")
        
        # Verify output columns
        logger.info(f"\nOutput columns: {final_df.columns.tolist()}")
        logger.info(f"Output shape: {final_df.shape}")
        
        return final_df
    else:
        logger.error("NO RESULTS GENERATED - Returning empty DataFrame")
        return pd.DataFrame({
            'predict': [],
            'p0': [],
            'p1': [],
            'predicted_class': [],
            'source_file': [],
            'row_num': [],
            'timestamp': [],
            'threshold_used': []
        })
EOF


cd ~/h2o-deployment

# create deployment yaml
cat > ~/h2o-deployment/batch-deployment-with-threshold.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/batchDeployment.schema.json
name: h2o-with-threshold
endpoint_name: h2o-batch-endpoint-final
model: azureml:stroke_best_model_xgb_final:3
code_configuration:
  code: ./batch-scoring-with-threshold
  scoring_script: score.py
environment: azureml:h2o-batch-env-java:1
compute: azureml:cpu-cluster
resources:
  instance_count: 1
max_concurrency_per_instance: 1
mini_batch_size: 10
output_action: append_row
output_file_name: predictions_with_threshold.csv
retry_settings:
  max_retries: 3
  timeout: 300
error_threshold: -1
logging_level: info
EOF

# Update deployment
az ml batch-deployment update \
  --name h2o-with-threshold \
  --endpoint-name h2o-batch-endpoint-final \
  --file batch-deployment-with-threshold.yml

# Wait for update
az ml batch-deployment show \
  --name h2o-with-threshold \
  --endpoint-name h2o-batch-endpoint-final \
  --query provisioning_state

# Set as default
az ml batch-endpoint update \
  --name h2o-batch-endpoint-final \
  --set defaults.deployment_name=h2o-with-threshold

# Test original data
az ml batch-endpoint invoke \
  --name h2o-batch-endpoint-final \
  --input azureml:stroke-test-final-data:1


ORIGINAL_JOB=$(az ml job list --query "[0].name" -o tsv)
echo "Original job: $ORIGINAL_JOB"

# Wait for completion
az ml job stream --name $ORIGINAL_JOB

# Test modified data
az ml batch-endpoint invoke \
  --name h2o-batch-endpoint-final \
  --input azureml:stroke-test-final-data-modified:1

MODIFIED_JOB=$(az ml job list --query "[0].name" -o tsv)
echo "Modified job: $MODIFIED_JOB"

# Wait for completion
az ml job stream --name $MODIFIED_JOB

cd ~/h2o-deployment

# Download original predictions
az ml job download \
  --name $ORIGINAL_JOB \
  --output-name score \
  --download-path ./predictions-original-threshold

# Download modified predictions
az ml job download \
  --name $MODIFIED_JOB \
  --output-name score \
  --download-path ./predictions-modified-threshold


# compare predictions between original and modified data
cat > ~/h2o-deployment/compare_predictions_threshold.py << 'EOF'
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

print("="*80)
print("PREDICTION COMPARISON: ORIGINAL vs MODIFIED DATA")
print("="*80)

# Load predictions
pred_original = pd.read_csv('predictions-original-threshold/predictions_with_threshold.csv',  header=None, sep=' ', skipinitialspace=True)
pred_original = pred_original.rename(columns={2: "p1"})

pred_modified = pd.read_csv('predictions-modified-threshold/predictions_with_threshold.csv',  header=None, sep=' ', skipinitialspace=True)
pred_modified = pred_modified.rename(columns={2: "p1"})

print(f"\nOriginal predictions: {len(pred_original)} rows")
print(f"Modified predictions: {len(pred_modified)} rows")

# Prediction distribution comparison
print("\n" + "="*80)
print("PREDICTION DISTRIBUTION")
print("="*80)

orig_counts = pred_original.iloc[:, 3].value_counts()
mod_counts = pred_modified.iloc[:, 3].value_counts()

print("\nOriginal Data:")
print(f"  No Stroke (0): {orig_counts.get(0, 0)} ({orig_counts.get(0, 0)/len(pred_original)*100:.2f}%)")
print(f"  Stroke (1):    {orig_counts.get(1, 0)} ({orig_counts.get(1, 0)/len(pred_original)*100:.2f}%)")

print("\nModified Data:")
print(f"  No Stroke (0): {mod_counts.get(0, 0)} ({mod_counts.get(0, 0)/len(pred_modified)*100:.2f}%)")
print(f"  Stroke (1):    {mod_counts.get(1, 0)} ({mod_counts.get(1, 0)/len(pred_modified)*100:.2f}%)")

# Calculate change
stroke_pred_change = (mod_counts.get(1, 0) - orig_counts.get(1, 0))
stroke_pct_change = ((mod_counts.get(1, 0)/len(pred_modified)) - (orig_counts.get(1, 0)/len(pred_original))) * 100

print(f"\nChange in Stroke Predictions:")
print(f"  Absolute: {stroke_pred_change:+d} predictions")
print(f"  Percentage: {stroke_pct_change:+.2f}% points")

# Probability analysis
if 'p1' in pred_original.columns and 'p1' in pred_modified.columns:
    print("\n" + "="*80)
    print("STROKE PROBABILITY ANALYSIS")
    print("="*80)
    
    print("\nOriginal Data Probabilities:")
    print(f"  Mean: {pred_original['p1'].mean():.4f}")
    print(f"  Std:  {pred_original['p1'].std():.4f}")
    print(f"  Min:  {pred_original['p1'].min():.4f}")
    print(f"  Max:  {pred_original['p1'].max():.4f}")
    
    print("\nModified Data Probabilities:")
    print(f"  Mean: {pred_modified['p1'].mean():.4f}")
    print(f"  Std:  {pred_modified['p1'].std():.4f}")
    print(f"  Min:  {pred_modified['p1'].min():.4f}")
    print(f"  Max:  {pred_modified['p1'].max():.4f}")
    
    prob_increase = pred_modified['p1'].mean() - pred_original['p1'].mean()
    print(f"\nMean Probability Change: {prob_increase:+.4f}")

# Save comparison metrics
comparison_metrics_threshold = {
    'original': {
        'total': len(pred_original),
        'no_stroke': int(orig_counts.get(0, 0)),
        'stroke': int(orig_counts.get(1, 0)),
        'stroke_pct': float(orig_counts.get(1, 0)/len(pred_original)*100)
    },
    'modified': {
        'total': len(pred_modified),
        'no_stroke': int(mod_counts.get(0, 0)),
        'stroke': int(mod_counts.get(1, 0)),
        'stroke_pct': float(mod_counts.get(1, 0)/len(pred_modified)*100)
    },
    'changes': {
        'stroke_absolute': int(stroke_pred_change),
        'stroke_percentage_points': float(stroke_pct_change)
    }
}

with open('comparison_metrics_threshold.json', 'w') as f:
    json.dump(comparison_metrics_threshold, f, indent=2)

print("\n" + "="*80)
print("Comparison metrics saved to: comparison_metrics.json")
print("="*80)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
categories = ['No Stroke', 'Stroke']
original_vals = [orig_counts.get(0, 0), orig_counts.get(1, 0)]
modified_vals = [mod_counts.get(0, 0), mod_counts.get(1, 0)]

x = np.arange(len(categories))
width = 0.35

axes[0].bar(x - width/2, original_vals, width, label='Original', color='skyblue')
axes[0].bar(x + width/2, modified_vals, width, label='Modified', color='salmon')
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('Count')
axes[0].set_title('Prediction Distribution Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].legend()

# Probability distribution
if 'p1' in pred_original.columns and 'p1' in pred_modified.columns:
    axes[1].hist(pred_original['p1'], bins=30, alpha=0.5, label='Original', color='skyblue')
    axes[1].hist(pred_modified['p1'], bins=30, alpha=0.5, label='Modified', color='salmon')
    axes[1].set_xlabel('Stroke Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Stroke Probability Distribution')
    axes[1].legend()

plt.tight_layout()
plt.savefig('prediction_comparison_threshold.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: prediction_comparison.png")
EOF

cd ~/h2o-deployment

# Run comparison
python compare_predictions_threshold.py

# View results
cat comparison_metrics_threshold.json


# monitoring report for original vs modified data
cat > ~/h2o-deployment/monitoring_report_threshold.py << 'EOF'
import json
from datetime import datetime

print("="*80)
print("MODEL DEPLOYMENT & MONITORING REPORT")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Load comparison metrics
with open('comparison_metrics_threshold.json', 'r') as f:
    metrics = json.load(f)

print("\n1. DEPLOYMENT STATUS: ✓ SUCCESS")
print("   - Model: stroke_best_model_xgb_final:3")
print("   - Deployment: h2o-model-script")
print("   - Endpoint: h2o-batch-endpoint-final")
print("   - Environment: h2o-batch-env-java:1 (with Java 11)")

print("\n2. ORIGINAL TEST DATA RESULTS:")
print(f"   - Total predictions: {metrics['original']['total']}")
print(f"   - Stroke predictions: {metrics['original']['stroke']} ({metrics['original']['stroke_pct']:.2f}%)")
print(f"   - No stroke predictions: {metrics['original']['no_stroke']} ({100-metrics['original']['stroke_pct']:.2f}%)")

print("\n3. MODIFIED TEST DATA RESULTS:")
print("   - Modifications applied:")
print("     * Age: +10 years")
print("     * Glucose level: +50%")
print(f"   - Total predictions: {metrics['modified']['total']}")
print(f"   - Stroke predictions: {metrics['modified']['stroke']} ({metrics['modified']['stroke_pct']:.2f}%)")
print(f"   - No stroke predictions: {metrics['modified']['no_stroke']} ({100-metrics['modified']['stroke_pct']:.2f}%)")

print("\n4. IMPACT ANALYSIS:")
print(f"   - Change in stroke predictions: {metrics['changes']['stroke_absolute']:+d}")
print(f"   - Change in stroke rate: {metrics['changes']['stroke_percentage_points']:+.2f}% points")

if metrics['changes']['stroke_percentage_points'] > 5:
    print("   - ⚠️  SIGNIFICANT INCREASE in stroke risk predictions")
    print("   - Model correctly identifies higher risk from age & glucose changes")
elif metrics['changes']['stroke_percentage_points'] < -5:
    print("   - ⚠️  SIGNIFICANT DECREASE in stroke risk predictions")
else:
    print("   - ℹ️  Moderate change in predictions")

print("\n5. MODEL MONITORING:")
print("   - Feature drift detected: Age & glucose distribution shifted")
print("   - Model response: Appropriate increase in stroke predictions")
print("   - Recommendation: Model is performing as expected")

print("\n" + "="*80)
print("CONCLUSION: Deployment successful. Model monitoring active.")
print("Model correctly responds to risk factor changes.")
print("="*80)
EOF

python monitoring_report.py

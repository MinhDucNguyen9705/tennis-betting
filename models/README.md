# Tennis Betting Model - Quick Start Guide

## Prerequisites

```bash
pip install pyyaml numpy pandas scikit-learn joblib matplotlib xgboost
```

Or if using the virtual environment:
```bash
source ../venv/bin/activate
pip install pyyaml numpy pandas scikit-learn joblib matplotlib xgboost
```

## Configuration

Edit `config.yaml` to customize:
- Data paths and years
- Model hyperparameters
- Feature engineering settings
- Output locations

## Usage

### 1. Train, Evaluate, and Save Model (All-in-one)

```bash
cd models
python run.py --mode all
```

This will:
- Load training data (2019-2022)
- Train model with calibration (2023)
- Evaluate on test set (2024)
- Save model to `./results/tennis_model.joblib`
- Generate evaluation report

### 2. Train Only

```bash
python run.py --mode train
```

### 3. Evaluate Only (requires trained model)

```bash
python run.py --mode eval
```

### 4. Run Inference on New Data

```bash
python run.py --mode infer --input ../data_tennis_match_reduced/parquet/matches_2024.parquet
```

Or specify in config.yaml:
```yaml
inference:
  input_file: "../data_tennis_match_reduced/parquet/matches_2024.parquet"
```

Then run:
```bash
python run.py --mode infer
```

### 5. Use Custom Config File

```bash
python run.py --config my_custom_config.yaml --mode all
```

## Configuration File Structure

### Data Settings
```yaml
data:
  parquet_dir: "../data_tennis_match_reduced/parquet"
  train_years: [2019, 2020, 2021, 2022]
  calib_years: [2023]
  test_years: [2024]
```

### Model Selection
```yaml
model:
  type: "histgradient"  # or "xgboost"
  threshold: 0.5
```

### Hyperparameters
```yaml
model:
  histgradient:
    learning_rate: 0.05
    max_iter: 200
    max_depth: 6
    # ... more parameters
```

### Output Locations
```yaml
output:
  results_dir: "./results"
  model_path: "./results/tennis_model.joblib"
  eval_report: "./results/evaluation_report.txt"
```

## Output Files

After running, you'll find:

- `results/tennis_model.joblib` - Trained model bundle
- `results/evaluation_report.txt` - Performance metrics
- `predictions/predictions_*.csv` - Inference results

## Example Workflow

```bash
# 1. Train and evaluate
python run.py --mode all

# 2. Check results
cat results/evaluation_report.txt

# 3. Run inference on new data
python run.py --mode infer --input path/to/new_data.parquet

# 4. Check predictions
head predictions/predictions_new_data.csv
```

## Model Bundle Contents

The saved model includes:
- Trained model (with calibration if enabled)
- Feature list
- Classification threshold
- Configuration snapshot
- Training timestamp

## Tips

1. **Quick test**: Reduce years in config for faster iteration
2. **Hyperparameter tuning**: Modify config.yaml and re-run
3. **Multiple experiments**: Create separate config files
4. **Batch inference**: List multiple files in config
5. **Reproducibility**: Set `random_state` in config

## Troubleshooting

### Missing parquet files
Check that `parquet_dir` points to the correct directory with files like `matches_2019.parquet`

### Out of memory
Reduce the number of years or increase system RAM

### Feature errors
Ensure your data has the required columns (ID_1, ID_2, tournament_date, etc.)

### Import errors
Make sure all dependencies are installed and you're running from the `models/` directory

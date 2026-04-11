import os
import argparse
import logging
import pandas as pd
from pathlib import Path

# Assumes sys.path is setup correctly in Colab, or run with python -m experiments.{EXPERIMENT}.prepare_dataset
try:
    from src.config.paths import get_bronze_path, get_silver_path, get_gold_path
except ImportError:
    # Fallback for direct execution without project path setup
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config.paths import get_bronze_path, get_silver_path, get_gold_path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def prepare_dataset(dataset_name: str, force: bool = False):
    """
    Reads structured data from SILVER (or falls back to BRONZE), performs 
    cleaning/splitting, and writes ML-ready parquets/csvs to GOLD.
    """
    try:
        bronze_dir = get_bronze_path("tabular") / dataset_name
        silver_dir = get_silver_path("tabular") / dataset_name
        gold_dir = get_gold_path("tabular") / dataset_name
    except NameError:
        from src.config.paths import BRONZE, SILVER, GOLD
        bronze_dir = BRONZE / dataset_name
        silver_dir = SILVER / dataset_name
        gold_dir = GOLD / dataset_name

    # Silver Fallback Logic
    if silver_dir.exists():
        source_dir = silver_dir
        logger.info(f"Using cleaned SILVER data from: {source_dir}")
        source_file = silver_dir / "wine_cleaned.csv"
    elif bronze_dir.exists():
        source_dir = bronze_dir
        logger.info(f"SILVER not found. Falling back to raw BRONZE data from: {source_dir}")
        source_file = bronze_dir / "winequality-red.csv"
    else:
        logger.error(f"Neither SILVER nor BRONZE directory found for {dataset_name}.")
        return
        
    if gold_dir.exists() and not force:
        logger.info(f"Gold directory already exists at {gold_dir}. Use --force to overwrite.")
        return
        
    logger.info(f"Starting preparation for {dataset_name}...")
    logger.info(f"Reading from: {source_file}")
    logger.info(f"Writing to: {gold_dir}")
    
    os.makedirs(gold_dir, exist_ok=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} samples")
    
    target_col = 'quality'
    # Shift quality from 3-8 to 0-5 for XGBoost
    df[target_col] = df[target_col] - df[target_col].min()

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/Val/Test Split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp) # 0.25 x 0.8 = 0.2
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val_scaled.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_parquet(gold_dir / "train.parquet", index=False)
    val_df.to_parquet(gold_dir / "val.parquet", index=False)
    test_df.to_parquet(gold_dir / "test.parquet", index=False)
    
    logger.info(f"Saved splits to GOLD: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BRONZE tabular dataset to GOLD.")
    parser.add_argument("--dataset", type=str, default="wine", help="Name of the dataset in BRONZE layer")
    parser.add_argument("--force", action="store_true", default=True, help="Overwrite existing GOLD directory")
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.force)

import pandas as pd
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config.paths import get_bronze_path, get_silver_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def clean_wine_data():
    bronze_dir = get_bronze_path("tabular") / "wine"
    silver_dir = get_silver_path("tabular") / "wine"
    
    bronze_file = bronze_dir / "winequality-red.csv"
    if not bronze_file.exists():
        logger.error(f"Bronze file not found: {bronze_file}")
        sys.exit(1)
        
    logger.info(f"Reading raw datalake CSV from Bronze: {bronze_file}")
    
    # The UCI Wine CSV is delimited by semicolons!
    df = pd.read_csv(bronze_file, sep=';')
    logger.info(f"Extracted shape: {df.shape}")
    
    # ---------------------------------------------------------
    # SILVER CLEANING RULES
    # ---------------------------------------------------------
    logger.info("Applying Silver schema standardization...")
    
    # 1. Clean column names (remove spaces, lowercase)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # 2. Drop NA rows
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} null rows.")
    
    # 3. Simulate casting/canonicalization
    for col in df.columns:
        if col != 'quality':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop enforced coercions that failed
    df = df.dropna()
    
    # 4. Save to Silver
    os.makedirs(silver_dir, exist_ok=True)
    silver_file = silver_dir / "wine_cleaned.csv"
    df.to_csv(silver_file, index=False)
    
    logger.info(f"Silver cleaning complete! Saved {df.shape} to {silver_file}")

if __name__ == "__main__":
    clean_wine_data()

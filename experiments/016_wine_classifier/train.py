import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# LOCAL SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import yaml
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from src.config.paths import setup_mlflow, TRAINED

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def write_completion_marker(config: dict, best_acc: float, duration: float, success: bool, error: str = None):
    marker = {
        "experiment": config["experiment"]["name"],
        "completed_at": datetime.now().isoformat(),
        "success": success,
        "duration_seconds": round(duration, 1),
        "best_val_acc": round(best_acc, 4) if best_acc else None,
        "model": config["model"]["algorithm"],
        "error": error,
    }
    marker_path = Path("completed.json")
    marker_path.write_text(json.dumps(marker, indent=2))
    logger.info("Completion marker written to %s", marker_path)

def load_data(config: dict):
    from src.config.paths import get_gold_path
    gold_dir = get_gold_path("tabular") / config["data"]["dataset"]
    
    train_df = pd.read_parquet(gold_dir / "train.parquet")
    test_df = pd.read_parquet(gold_dir / "test.parquet")
    
    target_col = config["data"]["target_column"]
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def get_model(config: dict):
    algo = config["model"]["algorithm"]
    params = config["model"].get("params", {})
    if algo == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    raise NotImplementedError(f"Algorithm {algo} not supported")

def main():
    config = load_config()
    start_time = time.time()
    best_acc = 0

    try:
        mlflow = setup_mlflow()
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

        X_train, X_test, y_train, y_test = load_data(config)
        model = get_model(config)

        with mlflow.start_run(run_name=config["mlflow"].get("run_name")):
            mlflow.log_params({
                "model": config["model"]["algorithm"],
                **config["model"].get("params", {}),
                "dataset": config["data"]["dataset"],
            })

            logger.info("Training...")
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            best_acc = test_acc

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

            mlflow.log_metrics({
                "train_acc": train_acc,
                "test_acc": test_acc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            })

            logger.info(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            save_dir = TRAINED / config["experiment"]["name"]
            save_dir.mkdir(parents=True, exist_ok=True)
            mlflow.sklearn.log_model(model, "model")

            report = classification_report(y_test, model.predict(X_test))
            logger.info(f"Classification Report:\n{report}")
            mlflow.log_text(report, "classification_report.txt")

        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=True)

    except Exception as e:
        duration = time.time() - start_time
        write_completion_marker(config, best_acc, duration, success=False, error=str(e))
        raise

if __name__ == "__main__":
    main()

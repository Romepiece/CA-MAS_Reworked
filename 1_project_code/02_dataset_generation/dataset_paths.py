from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DATA_DIR = OUTPUTS_DIR / "02_gee_collect" / "data_v2"
VALIDATION_OUTPUT_DIR = OUTPUTS_DIR / "03_validate_daily_stacks"
DATASET_OUTPUT_DIR = OUTPUTS_DIR / "04_dataset_loader"
DATASET_PATH = DATASET_OUTPUT_DIR / "dataset_pixels.h5"
DATASET_RANDOM_PATH = DATASET_OUTPUT_DIR / "dataset_pixels_random.h5"
ML_BASELINE_DIR = OUTPUTS_DIR / "06_train_ml_baseline"
CA_BASELINE_DIR = OUTPUTS_DIR / "07_train_ca_baseline" / "one_step"
CA_ROLLOUT_DIR = OUTPUTS_DIR / "07_train_ca_baseline" / "rollout"
COMPARISON_DIR = OUTPUTS_DIR / "08_compare_ca_ml"
HYBRID_BASELINE_DIR = OUTPUTS_DIR / "09_train_hybrid_ca_ml"
ML_INSIDE_CA_DIR = OUTPUTS_DIR / "10_train_ml_inside_ca"
CASES_CONFIG_PATH = SRC_DIR / "cases_config.json"


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return PROJECT_ROOT / path
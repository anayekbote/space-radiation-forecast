from pathlib import Path

# Base project root dynamically resolved across Linux, Windows, and macOS
BASE_DIR = Path(__file__).resolve().parent.parent

# Subdirectories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
SRC_DIR = BASE_DIR / "src"

# Auto-provision directories if not present
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=" * 65)
    print("TEAM ANTARIKSH · दृष्टि (DRISHTI) | SYSTEM CONFIGURATION")
    print("=" * 65)
    print(f"Base Directory       : {BASE_DIR}")
    print(f"Raw Data Path        : {RAW_DATA_DIR}")
    print(f"Processed Data Path  : {PROCESSED_DATA_DIR}")
    print(f"Models Directory     : {MODELS_DIR}")
    print(f"Plots Directory      : {PLOTS_DIR}")
    print("=" * 65)

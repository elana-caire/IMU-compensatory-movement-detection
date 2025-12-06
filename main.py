import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    print(f"\n Running: {script_path}\n")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}. ")
        sys.exit(result.returncode)
    print(f"Finished: {script_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline launcher")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["prepare", "train", "all"],
        help="Which stage to run: data preparation, model training, or everything.",
    )

    args = parser.parse_args()

    root = Path(__file__).parent

    data_prep_script = root / "data_preparation.py"
    model_training_script = root / "model_training.py"

    if args.stage in ("prepare", "all"):
        run_script(str(data_prep_script))

    if args.stage in ("train", "all"):
        run_script(str(model_training_script))

    print("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()

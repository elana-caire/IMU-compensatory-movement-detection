import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path):
    print(f"\nRunning: {script_path}\n")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}.")
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

    parser.add_argument(
        "--training-type",
        type=str,
        choices=["task_agnostic", "task_specific"],
        default="task_agnostic",
        help="Which training script to run.",
    )

    args = parser.parse_args()

    root = Path(__file__).parent

    data_prep_script = root / "data_preparation.py"

    training_scripts = {
        "task_agnostic": root / "train_models_task_agnostic.py",
        "task_specific": root / "train_model_task_specific.py",
    }

    if args.stage in ("prepare", "all"):
        run_script(str(data_prep_script))

    if args.stage in ("train", "all"):
        run_script(str(training_scripts[args.training_type]))

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

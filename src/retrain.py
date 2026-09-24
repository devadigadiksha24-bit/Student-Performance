import subprocess


print("==============================")
print("AUTOMATIC RETRAINING")
print("==============================")


print("\nStep 1: Preprocessing data")

subprocess.run(
    ["python", "src/preprocess.py"],
    check=True
)


print("\nStep 2: Training models")

subprocess.run(
    ["python", "src/train.py"],
    check=True
)


print("\nStep 3: Model evaluation")

subprocess.run(
    ["python", "src/evaluate.py"],
    check=True
)


print("\nRetraining pipeline completed successfully.")
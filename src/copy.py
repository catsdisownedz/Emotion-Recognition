from pathlib import Path

run_dir = Path("/Users/malakmaher/Documents/GitHub/Emotion-Recognition/notebooks/models/run_2025-12-13_04-53-19")

print("Run dir exists:", run_dir.exists())
print("\nFiles directly under run_dir:")
for p in sorted(run_dir.glob("*")):
    print(" -", p.name)

print("\nSearch for model and json inside run_dir:")
for p in sorted(run_dir.rglob("*")):
    if p.suffix in [".h5", ".keras", ".json"]:
        print(" -", p)

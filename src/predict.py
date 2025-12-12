# src/predict.py
import json
import argparse

from src.predictor import EmotionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to an audio file (.wav/.mp3/.flac)")
    parser.add_argument("--run_dir", default=None, help="Optional: specific run_ folder to use")
    args = parser.parse_args()

    predictor = EmotionPredictor(run_dir=args.run_dir)
    result = predictor.predict(args.file)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from extract_metadata import extract_metadata
from feature_engineering import metadata_to_features

BASE_DIR = "dataset"
OUTPUT_FILE = "results/features.csv"

def process_folder(folder_name, label):
    rows = []
    folder_path = os.path.join(BASE_DIR, folder_name)

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)
            metadata = extract_metadata(path)
            features = metadata_to_features(metadata, label)
            rows.append(features)

    return rows

def main():
    os.makedirs("results", exist_ok=True)

    data = []
    data += process_folder("real", 0)
    data += process_folder("ai_generated", 1)

    print("Total samples collected:", len(data))

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Feature dataset saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
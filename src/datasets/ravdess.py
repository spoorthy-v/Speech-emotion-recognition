from pathlib import Path
import pandas as pd


EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def create_ravdess_metadata(data_dir):
    data_dir = Path(data_dir)

    records = []

    for audio_file in sorted(data_dir.rglob("*.wav")):
        parts = audio_file.stem.split("-")

        if len(parts) != 7:
            continue

        modality = int(parts[0])
        vocal_channel = int(parts[1])
        emotion_id = int(parts[2])
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor_id = int(parts[6])

        records.append(
            {
                "file_path": str(audio_file),
                "modality": modality,
                "vocal_channel": vocal_channel,
                "emotion_id": emotion_id,
                "emotion": EMOTION_MAP[emotion_id],
                "intensity": intensity,
                "statement": statement,
                "repetition": repetition,
                "actor_id": actor_id,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    data_dir = "data/raw/RAVDESS"
    output_file = "data/metadata/ravdess_metadata.csv"

    df = create_ravdess_metadata(data_dir)

    df.to_csv(output_file, index=False)

    print(f"Created metadata for {len(df)} audio files.")
    print(f"Saved to: {output_file}")

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts().sort_index())

    print("\nActor distribution:")
    print(df["actor_id"].value_counts().sort_index())
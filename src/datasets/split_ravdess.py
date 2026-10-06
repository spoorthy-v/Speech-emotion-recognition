from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def create_speaker_independent_split(metadata_path, output_path):
    df = pd.read_csv(metadata_path)

    # Get all unique actors
    actors = sorted(df["actor_id"].unique())

    # Check that RAVDESS contains 24 actors
    if len(actors) != 24:
        raise ValueError(
            f"Expected 24 actors, but found {len(actors)} actors."
        )

    # ---------------------------------------------------------
    # Step 1: Split actors into 16 train actors and 8 temp actors
    # ---------------------------------------------------------
    train_actors, temp_actors = train_test_split(
        actors,
        test_size=8,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # Step 2: Split the remaining 8 actors into
    # 4 validation actors and 4 test actors
    # ---------------------------------------------------------
    val_actors, test_actors = train_test_split(
        temp_actors,
        test_size=4,
        random_state=RANDOM_STATE,
    )

    # Sort actor IDs for reproducibility/readability
    train_actors = sorted(train_actors)
    val_actors = sorted(val_actors)
    test_actors = sorted(test_actors)

    # ---------------------------------------------------------
    # Assign split labels
    # ---------------------------------------------------------
    df["split"] = df["actor_id"].apply(
        lambda actor: (
            "train"
            if actor in train_actors
            else "validation"
            if actor in val_actors
            else "test"
        )
    )

    # ---------------------------------------------------------
    # Save split metadata
    # ---------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    # ---------------------------------------------------------
    # Print split information
    # ---------------------------------------------------------
    print("Train actors:", train_actors)
    print("Validation actors:", val_actors)
    print("Test actors:", test_actors)

    print("\nNumber of actors:")
    print("Train:", len(train_actors))
    print("Validation:", len(val_actors))
    print("Test:", len(test_actors))

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nEmotion distribution by split:")
    print(
        pd.crosstab(
            df["split"],
            df["emotion"]
        )
    )


if __name__ == "__main__":
    create_speaker_independent_split(
        "data/metadata/ravdess_metadata.csv",
        "data/metadata/ravdess_splits.csv"
    )
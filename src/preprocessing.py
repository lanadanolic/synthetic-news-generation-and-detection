# Functions for loading, cleaning, splitting, and saving the dataset
# %%
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# %%
def clean_text(text):

    # Cleans the news article text

    if pd.isna(text):
        return ""

    text = str(text)

    # Removes URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Removes Reuters labels
    text = re.sub(
        r"\bReuters\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Removes extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
# %%
def load_isot_data(fake_path, true_path):

    # Loads Fake.csv and True.csv

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # 0 = real, 1 = fake
    true_df["label"] = 0
    fake_df["label"] = 1

    true_df["original_type"] = "real"
    fake_df["original_type"] = "fake"

    for frame in [true_df, fake_df]:

        frame["title"] = frame["title"].apply(
            clean_text
        )

        frame["text"] = frame["text"].apply(
            clean_text
        )

        frame["full_text"] = (
            frame["title"]
            + ". "
            + frame["text"]
        )

    columns = [
        "title",
        "text",
        "full_text",
        "label",
        "original_type"
    ]

    true_df = true_df[columns]
    fake_df = fake_df[columns]

    # Combines real and fake news
    data = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    # Removes empty and very short texts
    data = data[
        data["full_text"].str.len() > 50
    ].copy()

    # Removes duplicates
    data = data.drop_duplicates(
        subset=["full_text"]
    ).reset_index(drop=True)

    # Assigns a unique ID to each article
    data["source_id"] = [
        f"isot_{i:05d}"
        for i in range(len(data))
    ]

    return data
# %%
def split_dataset(
    data,
    test_size=0.20,
    random_state=42
):
    # Splits the data into training and test sets

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df
# %%
def save_splits(
    train_df,
    test_df,
    output_dir
):
    # Saves the training and test sets as CSV files

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )
# %%

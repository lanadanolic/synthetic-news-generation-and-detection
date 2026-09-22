# %% Functions for generating synthetic news variants using an LLM
import time
import pandas as pd

from pydantic import BaseModel
from typing import Literal

from google.genai import types

# %%
class SyntheticVariant(BaseModel):
    manipulation_type: Literal[
        "fact_change",
        "clickbait",
        "tone_shift"
    ]

    usable: bool
    title: str
    text: str
    change_summary: str
# %% 
def create_generation_prompt(title, text, max_chars=4000):

    # Creates a prompt for generating three controlled synthetic variants of a news article
    
    text = str(text)[:max_chars]

    prompt = f"""
        You are creating controlled synthetic data for an academic
        research project on manipulated-news detection.

        The generated examples will ONLY be used to train and evaluate
        a defensive classifier.

        ORIGINAL AUTHENTIC NEWS ARTICLE

        TITLE:
        {title}

        TEXT:
        {text}


        Create exactly THREE synthetic variants.

        1. fact_change

        Change exactly ONE low-risk factual detail from the article.

        Examples of allowed changes:
        - number
        - quantity
        - date
        - year
        - non-sensitive location

        Keep everything else as close as possible to the original article.

        Do NOT invent:
        - crimes
        - accusations
        - medical claims
        - election claims
        - financial accusations
        - harmful claims about identifiable people

        If a safe factual modification cannot be made, set usable to false.


        2. clickbait

        Rewrite the TITLE into a more exaggerated, attention-grabbing clickbait headline.

        Keep the underlying facts unchanged.

        The body should remain mostly consistent with the original article.

        Do not introduce new factual claims.


        3. tone_shift

        Keep all important facts unchanged.

        Rewrite the article using a more dramatic, emotional and alarmist tone. Do not introduce new factual claims.

        GENERAL RULES:

        - Return exactly one variant for each manipulation type.
        - Keep approximately the same amount of information.
        - Do not mention that the text is synthetic.
        - Do not add explanations outside the requested output.
    """

    return prompt
# %%
def generate_variants(
    client,
    model_name,
    title,
    text,
    max_attempts=4
):

    # Calls Gemini and returns three structured synthetic variants
  
    prompt = create_generation_prompt(
        title,
        text
    )

    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    response_mime_type="application/json",
                    response_schema=SyntheticBatch
                )
            )

            if response.parsed:
                return response.parsed

            return SyntheticBatch.model_validate_json(
                response.text
            )

        except Exception as error:

            print(
                f"Attempt {attempt} failed:",
                str(error)[:200]
            )

            if attempt < max_attempts:
                time.sleep(
                    2 ** (attempt - 1)
                )

    return None
# %%
def generate_synthetic_dataset(
    source_df,
    client,
    model_name
):
    # Generates a synthetic dataset from real training news articles

    records = []

    total = len(source_df)

    for i, row in source_df.iterrows():

        print(
            f"Attempt {i + 1}/{total}"
        )

        batch = generate_variants(
            client=client,
            model_name=model_name,
            title=row["title"],
            text=row["text"]
        )

        if batch is None:
            print("Attempts to generate variants failed. Skipping this article.")
            continue

        for variant in batch.variants:

            if not variant.usable:
                continue

            generated_title = (
                variant.title.strip()
            )

            generated_text = (
                variant.text.strip()
            )

            if len(generated_text) < 50:
                continue

            full_text = (
                generated_title
                + ". "
                + generated_text
            )

            records.append({
                "source_id": row["source_id"],
                "title": generated_title,
                "text": generated_text,
                "full_text": full_text,
                "label": 1,
                "original_type": "synthetic",
                "manipulation_type":
                    variant.manipulation_type,
                "change_summary":
                    variant.change_summary,
                "synthetic": True
            })

    return pd.DataFrame(records)
# %%

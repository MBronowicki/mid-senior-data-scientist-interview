import spacy
import re
import emoji
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def clean_data(df):
    """Perform basic data cleaning like missing values, duplicates"""
    df = df.drop_duplicates(subset='id') # Remove duplicates
    df = df.dropna()
    df.set_index('id', inplace=True)
    return df

def convert_columns_dtypes(df):
    """Convert column types for performance"""
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['is_edited'] = df['is_edited'].astype('int32')
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category')
    return df

# Note: this function do not remove all non-english text and needs further investigation
def count_and_remove_non_english_rows(df, column="company_description"):
    """Removes rows with language different then English"""
    non_english_indices = []  # To store indices of non-English rows
    non_english_count = 0

    # Iterate through each row to detect the language
    for idx, text in df[column].dropna().items():
        try:
            if detect(text) != 'en':  # If detected language is not English
                non_english_count += 1
                non_english_indices.append(idx)  # Append the index of non-English rows
                # print(f"Removing: {text}")
        except LangDetectException:
            # If language detection fails for some reason, we ignore that row
            non_english_indices.append(idx)  # Treat failed detections as non-English

    # print(non_english_indices)

    # Remove non-English rows by their indices
    df_cleaned = df.drop(index=non_english_indices)

    print(f"Number of non-English rows: {non_english_count}")
    print(f"Shape of cleaned data: {df_cleaned.shape}")

    # Return the count of non-English rows and the cleaned DataFrame
    return df_cleaned


def preprocess_text(text):
    """Function to preprocess text using spaCy's stopwords"""
    # Lowercase the text
    text = text.lower()
    # print(text)

    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = emoji.replace_emoji(text, replace="") # Remove emojis
    text = re.sub(r'\S+@\S+', '', text) # Remove email addresses

    # Remove special characters and numbers (only keep alphabetic characters)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Apply spaCy NLP pipeline (tokenization, lemmatization, etc.)
    doc = nlp(text)
    
    # Remove stopwords and non-alphabetic tokens, apply lemmatization
    cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    
    return cleaned_text

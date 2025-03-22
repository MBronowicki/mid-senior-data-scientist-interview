import pandas as pd

def extract_features(df, column='company_description'):
    """Extract some features from text"""
    df["text_lenght"] = df[column].apply(lambda x: len(x.split()))
    return df

def feature_engineering_datetime(df, datetime_column='created_at'):
    """
    Simple feature engineering for the datetime column.
    Extract year, month, day of the week, and whether the date is a weekend
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    df['is_weekend'] = (df[datetime_column].dt.weekday >=5).astype(int) # turn into integer to save sparse matrix later on
    df['hour'] = df[datetime_column].dt.hour
    return df

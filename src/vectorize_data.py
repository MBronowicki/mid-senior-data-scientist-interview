import numpy as np
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text_with_tfidf(df):
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Transform the text column into a TF-IDF matrix
    X_text = tfidf_vectorizer.fit_transform(df['company_description'])
    # If you want to inspect the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    return X_text

def save_sparse_matrix(df):
    # Combine TF-IDF features with other numerical features (assuming 'df' has numerical columns)
    X_other = df[['year','month', 'day', 'is_weekend',
                            'day_of_week', 'hour']].values  # Extract numerical columns
    
    # Create tfidf matrix
    X_text = vectorize_text_with_tfidf(df)
    # Convert numerical features to a sparse matrix
    # X_other_sparse = csr_matrix(X_other)
    final_features = hstack([X_text, X_other])

    # Save the sparse matrix to a file
    save_npz('./data/sparse_matrix.npz', final_features)
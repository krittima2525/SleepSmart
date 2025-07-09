import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

filepath = '../data/Sleep_health_and_lifestyle_dataset.csv'
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Example preprocessing steps
    # Transform NaN values in 'Sleep Disorder' column to "Normal"
    df['Sleep Disorder'] = df['Sleep Disorder'].where(df['Sleep Disorder'].notna(), "Normal")
    df = df.dropna()  # Remove missing values
    X = df.drop('Sleep Disorder', axis=1)  # Features
    y = df['Sleep Disorder']  # Target variable
    X = pd.get_dummies(X)  # Convert categorical columns to numeric
    return X, y

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
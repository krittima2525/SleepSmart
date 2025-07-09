import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.preprocessing import StandardScaler

# Path to the dataset
# Update the file path as necessary
file_path = '../data/Sleep_health_and_lifestyle_dataset.csv'

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Transform NaN values in 'Sleep Disorder' column to "Normal"
    df['Sleep Disorder'] = df['Sleep Disorder'].where(df['Sleep Disorder'].notna(), "Normal")
    df = df.dropna()  # Remove missing values
    X = df.drop('Sleep Disorder', axis=1)  # Features
    y = df['Sleep Disorder']  # Target variable
    X = pd.get_dummies(X)  # Convert categorical columns to numeric
    return X, y

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    df = load_data(file_path)
    X, y = preprocess_data(df)

    # Apply SMOTE after encoding
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    #print("Class distribution after SMOTE:")
    #print(pd.Series(y_resampled).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   

    
    model = train_model(X_train, y_train)
    save_model(model, 'disorder_detection_model_LogisticsRegression.pkl')
    
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
  
    # Calculate metrics
    Precision = precision_score(y_test, y_pred, average='weighted')
    Sensitivity_recall = recall_score(y_test, y_pred, average='weighted')
    if len(model.classes_) == 2:
        Specificity = recall_score(y_test, y_pred, pos_label=model.classes_[0])
    else:
        Specificity = None
    F1_score = f1_score(y_test, y_pred, average='weighted')
    Accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", Accuracy)
    print("Precision: ", Precision)
    print("Sensitivity_recall: ", Sensitivity_recall)
    print("Specificity: ", Specificity)
    print("F1_score: ", F1_score)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_LogisticRegression.png')

if __name__ == "__main__":
    main()
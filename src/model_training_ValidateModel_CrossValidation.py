import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

file_path = '../data/Sleep_health_and_lifestyle_dataset.csv'

def load_data(file_path):
    import os
    #print("Loading file:", os.path.abspath(file_path))
    df = pd.read_csv(file_path)
    #print("Columns:", df.columns)
    return df

def preprocess_data(df):
    df['Sleep Disorder'] = df['Sleep Disorder'].where(df['Sleep Disorder'].notna(), "Normal")
    df = df.dropna()
    if 'Person ID' in df.columns:
        df = df.drop('Person ID', axis=1)
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y, y_encoded, le

def get_model(model_name):
    if model_name == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'LogisticRegression':
        return LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == 'SVC':
        return SVC(kernel='rbf', probability=True, random_state=42)
    elif model_name == 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        if xgb_available:
            return XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
        else:
            raise ImportError("XGBoost is not installed.")
    elif model_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def save_model(model, filename):
    joblib.dump(model, filename)

results = {}

# Store metrics for each model
metrics = {
    'Model': [],
    'Metric': [],
    'Score': []
}

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        top_n = 10  # Show top 10 features
        print(f"\nTop {top_n} features for {model_name}:")
        for i in range(top_n):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices][:top_n], y=[feature_names[i] for i in indices[:top_n]])
        plt.title(f"Top {top_n} Feature Importances for {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png')
        plt.close()
    else:
        print(f"Feature importance not available for {model_name}")

def main(model_name='RandomForest'):
    df = load_data(file_path)
    X, y, y_encoded, le = preprocess_data(df)

    smote = SMOTE(random_state=42)
    # Always use y_encoded for XGBoost
    if model_name == 'XGBoost':
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    else:
        X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = get_model(model_name)

    # For XGBoost, y_train and y_test must be numeric
    if model_name == 'XGBoost':
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    results[model_name] = cv_scores.mean()

    model.fit(X_train, y_train)
    save_model(model, f'disorder_detection_model_{model_name}.pkl')

    # Plot feature importance for supported models
    plot_feature_importance(model, X.columns, model_name)
    
    y_pred = model.predict(X_test)
    # If using XGBoost, decode predictions for reporting
    if model_name == 'XGBoost':
        y_test = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(y_pred)
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
  
    Precision = precision_score(y_test, y_pred, average='weighted')
    Sensitivity_recall = recall_score(y_test, y_pred, average='weighted')
    Accuracy = accuracy_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {Accuracy:.5f}")
    print(f"Precision: {Precision:.5f}")
    print(f"Sensitivity_recall: {Sensitivity_recall:.5f}")
    print(f"F1_score: {F1_score:.5f}")


    # Collect cross-validation scores for each metric
    scoring_metrics = {
        'Accuracy': 'accuracy',
        'Precision': 'precision_weighted',
        'Recall': 'recall_weighted',
        'F1': 'f1_weighted'
    }
    for metric_name, scoring in scoring_metrics.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            for score in cv_scores:
                metrics['Model'].append(model_name)
                metrics['Metric'].append(metric_name)
                metrics['Score'].append(score)
        except Exception as e:
            print(f"Could not compute {metric_name} for {model_name}: {e}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix from {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')

if __name__ == "__main__":
    # List of all supported models
    model_names = [
        'RandomForest',
        'LogisticRegression',
        'SVC',
        'GradientBoosting',
        'XGBoost',
        'KNN',
        'DecisionTree'
    ]
    for model_name in model_names:
        print(f"\nRunning model: {model_name}")
        try:
            main(model_name)
        except Exception as e:
            print(f"Error running {model_name}: {e}")
    print("\nSummary of mean cross-validation accuracy:")
    for model, score in results.items():
        print(f"{model}: {score:.5f}")

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame(metrics)
    # Do NOT melt, just use metrics_df directly

    plt.figure(figsize=(12, 7))
    palette = sns.color_palette('Set2', n_colors=metrics_df['Metric'].nunique())
    sns.boxplot(
        x='Model',
        y='Score',
        hue='Metric',
        data=metrics_df,
        palette=palette,
        #width=0.5  # Reduce width for more space between boxes
    )
    sns.swarmplot(
        x='Model',
        y='Score',
        hue='Metric',
        data=metrics_df,
        dodge=True,
        alpha=0.7,
        size=5,
        edgecolor='auto',
        linewidth=0.5,
        marker='o',
        legend=False,
        palette=palette
    )
    plt.title('Model Performance Metrics by Model')
    plt.legend(title='Metric')
    plt.savefig('model_metrics_boxplot_by_model.png')
    #plt.show()
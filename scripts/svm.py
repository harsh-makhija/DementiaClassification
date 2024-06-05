import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

def load_data(file_path='/Users/harshmakhija/Desktop/CSC3432/data/processed.csv'):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def prepare_data(df):
    """
    Prepare the dataset for analysis by separating features and target and dropping unnecessary columns.

    Parameters:
    df (DataFrame): The DataFrame to be prepared.

    Returns:
    X (DataFrame): Features DataFrame.
    y (Series): Target variable series.
    """
    # Dropping columns that are not needed for the analysis
    df = df.drop(['Unnamed: 0', 'hand', 'MRI_ID', 'visit'], axis=1)

    # Separating the dataset into features (X) and target variable (y)
    X = df.drop('CDR', axis=1)
    y = df['CDR']
    return X, y

def encode_and_scale_features(X):
    """
    Encode categorical features and scale numerical features.

    Parameters:
    X (DataFrame): The features DataFrame.

    Returns:
    X_processed (DataFrame): The DataFrame with processed features.
    feature_names (Index): The names of the processed features.
    """
    # Encoding categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Scaling numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_scaled = scaler.fit_transform(X[numerical_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols)

    # Combining encoded and scaled features
    X_processed = pd.concat([X_scaled_df, X_encoded_df], axis=1)
    return X_processed, X_processed.columns

def perform_grid_search(X_train, y_train):
    """
    Perform a grid search to find the best hyperparameters for the SVC model.

    Parameters:
    X_train (DataFrame): The training feature set.
    y_train (Series): The training target set.

    Returns:
    GridSearchCV object: The GridSearchCV object after fitting.
    """
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }
    clf = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, cv=5, scoring='f1_weighted')
    clf.fit(X_train, y_train)
    return clf

def svc_classification(X, y, feature_names):
    """
    Classify the data using an SVC model, evaluate the model, and visualize the results.

    Parameters:
    X (DataFrame): The processed features.
    y (Series): The target variable.
    feature_names (Index): The names of the processed features.
    """
    # Adjusting SMOTE's k_neighbors based on class distribution
    smallest_class_count = y.value_counts().min()
    k_neighbors = min(smallest_class_count - 1, 5)  # Ensure k_neighbors is less than the smallest class count

    # Applying SMOTE for handling class imbalance
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Splitting the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled ,y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
    
    # Performing grid search to find the best parameters for SVC
    grid_search = perform_grid_search(X_train, y_train)
    best_svc = grid_search.best_estimator_
    
    # Fitting the model with the best parameters
    best_svc.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = best_svc.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report for SVC model:\n", classification_report(y_test, y_pred))

    # Analyzing feature importance
    perm_importance = permutation_importance(best_svc, X_test, y_test)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': perm_importance.importances_mean})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importances for SVC model:\n", importance_df)

    # Preparing and plotting the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.drop('support', axis=1, inplace=True)
    report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'], inplace=True)
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Classification Report for SVC Model')
    plt.ylabel('Score')
    plt.xlabel('CDR')
    plt.xticks(rotation=0)
    plt.show()

    # Computing and plotting the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=best_svc.classes_, columns=best_svc.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix for SVC Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    df = load_data()
    X, y = prepare_data(df)
    X_processed, feature_names = encode_and_scale_features(X)  # Capture the returned feature names
    svc_classification(X_processed, y, feature_names)  # Pass feature_names to the classification function

if __name__ == "__main__":
    main()

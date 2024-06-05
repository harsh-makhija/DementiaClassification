import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path = '/Users/harshmakhija/Desktop/CSC3432/data/processed.csv'):
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
    Prepare the dataset for analysis by separating features and target, and dropping unnecessary columns.

    Parameters:
    df (DataFrame): The DataFrame to be prepared.

    Returns:
    X (DataFrame): Features DataFrame.
    y (Series): Target variable series.
    """

    # Dropping unnecessary columns
    df = df.drop(['Unnamed: 0', 'hand', 'MRI_ID', 'visit'], axis=1)

    # Separating features and target
    X = df.drop('CDR', axis=1)
    y = df['CDR']
    return X, y

def encode_categorical_columns(X):
    """
    Encode categorical columns in the dataset.

    Parameters:
    X (DataFrame): The DataFrame containing the features.

    Returns:
    X (DataFrame): The DataFrame with encoded categorical features.
    feature_names (Index): The names of the features after encoding.
    """

    # Encoding categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate the encoded ones
    X = X.drop(categorical_cols, axis=1)
    X = pd.concat([X, X_encoded_df], axis=1)
    
    return X, X.columns  # Return the DataFrame and the new feature names

def perform_grid_search(X_train, y_train):
    """
    Perform a grid search to find the best hyperparameters for the RandomForestClassifier.

    Parameters:
    X_train (DataFrame): The training feature set.
    y_train (Series): The training target set.

    Returns:
    GridSearchCV object: The GridSearchCV object after fitting.
    """

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                       param_grid, cv=5, scoring='f1_weighted')
    clf.fit(X_train, y_train)
    return clf

def random_forest_classification(X, y, feature_names):
    """
    Classify the data using a Random Forest model, evaluate the model, and visualize the results.

    Parameters:
    X (DataFrame): The processed features.
    y (Series): The target variable.
    feature_names (Index): The names of the processed features.
    """

    # Applying SMOTE for handling class imbalance
    smote = SMOTE(k_neighbors=min(len(X) - 1, 2), random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Splitting the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
    
    # Performing grid search to find the best parameters for SVC
    grid_search = perform_grid_search(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # Fitting the model with the best parameters
    best_clf.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = best_clf.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report for Random Forests Model:\n", classification_report(y_test, y_pred))

    # Analyzing feature importance
    importances = best_clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importances for Random Forests Model:\n", importance_df)

    # Preparing and plotting the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Dropping support column and rows not needed for visualization
    report_df.drop('support', axis=1, inplace=True)
    report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'], inplace=True)

    # Plot
    report_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Random Forest Classification Report')
    plt.ylabel('Score')
    plt.xlabel('CDR')
    plt.xticks(rotation=0)
    plt.show()

    # Computing and plotting the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=best_clf.classes_, columns=best_clf.classes_)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def main():
    file_path = '/Users/harshmakhija/Desktop/CSC3432/data/processed.csv'
    df = load_data(file_path)
    X, y = prepare_data(df)
    X, feature_names = encode_categorical_columns(X)
    random_forest_classification(X, y, feature_names)

if __name__ == "__main__":
    main()

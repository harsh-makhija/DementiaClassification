import pandas as pd
from preprocessing import preprocess_data
from randomforests import random_forest_classification, encode_categorical_columns
from svm import svc_classification, encode_and_scale_features
from eda_script import exploratory_data_analysis
from kmeans_script import apply_kmeans,standardize_data, preprocess_data as kmeans_preprocess, plot_clusters, cluster_characterization

def main():

    # Set the pandas option to display all columns
    pd.set_option('display.max_columns', None)

# Set the pandas option to display a specific number of rows
    pd.set_option('display.max_rows', 100)

# Set the pandas option to display more characters in column width
    pd.set_option('display.max_colwidth', None)  # or some large number like 100

    # Preprocessing Data
    preprocess_data()

    # Load Processed Data
    file_path = '/Users/harshmakhija/Desktop/CSC3432/data/processed.csv'
    df = pd.read_csv(file_path)

    # Prepare Data for Random Forests and SVM
    X = df.drop(['CDR'], axis=1)
    y = df['CDR']

    # Random Forest Classification
    X_rf, feature_names_rf = encode_categorical_columns(X)
    random_forest_classification(X_rf, y, feature_names_rf)

    # SVM Classification
    X_svm, feature_names_svm = encode_and_scale_features(X)
    svc_classification(X_svm, y, feature_names_svm)

    exploratory_data_analysis(file_path)

  # K-means Clustering
    # Preprocess and standardize data for K-means
    df_kmeans = kmeans_preprocess(df)
    numerical_cols = ['age', 'delay', 'YOE', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    df_kmeans = standardize_data(df_kmeans, numerical_cols)

    # Apply K-means (assuming optimal_k is determined)
    optimal_k = 4  # Replace with the determined optimal number of clusters
    df_kmeans = apply_kmeans(df_kmeans, optimal_k)

    # Ensure to use df_kmeans for plotting and characterization
    plot_clusters(df_kmeans, 'Cluster')
    cluster_characterization(df_kmeans, 'Cluster')


if __name__ == "__main__":
    main()

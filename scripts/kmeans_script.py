
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data by applying one-hot encoding to categorical columns.

    Parameters:
    df (DataFrame): The DataFrame to preprocess.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
        
    # Identifying categorical columns for one-hot encoding
    categorical_cols = ['sex', 'hand', 'CDR']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # # Impute missing values
    # imputer = SimpleImputer(strategy='mean')
    # df['MMSE'] = imputer.fit_transform(df[['MMSE']])
    # df['SES'] = imputer.fit_transform(df[['SES']])

    return df

def standardize_data(df, numerical_cols):
    """
    Standardize numerical columns in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to standardize.
    numerical_cols (list): List of column names to standardize.

    Returns:
    DataFrame: The DataFrame with standardized numerical columns.
    """

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def determine_k(df):
    """
    Determine the optimal number of clusters using the elbow method.

    Parameters:
    df (DataFrame): The DataFrame to analyze.

    Displays:
    A plot showing the inertia for different numbers of clusters.
    """

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.show()

def apply_kmeans(df, k):
    """
    Apply K-means clustering to the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to cluster.
    k (int): The number of clusters to form.

    Returns:
    DataFrame: The original DataFrame with an additional 'Cluster' column.
    """

    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Apply K-means clustering to numeric data only
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_df)
    
    # Add cluster labels back to the original dataframe
    df['Cluster'] = cluster_labels
    return df

def main():
    file_path = '/path/to/your/dataset.csv'
    df = load_data(file_path)
    
    df = preprocess_data(df)
    
    numerical_cols = ['age', 'delay', 'YOE', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    df = standardize_data(df, numerical_cols)
    
    # Uncomment below line to determine the optimal number of clusters
    determine_k(df)
    
    optimal_k = 4  # Replace this with the determined optimal k
    df = apply_kmeans(df, optimal_k)

    print(df.head())

def plot_clusters(df, cluster_column, num_components=2):
    """
    Plot clusters using PCA for dimensionality reduction.

    Parameters:
    df (DataFrame): The DataFrame with cluster labels.
    cluster_column (str): The column name for cluster labels.
    num_components (int): Number of principal components for PCA.
    """

    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(df.select_dtypes(include=[np.number]))
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df[cluster_column])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Visualization with PCA')
    plt.colorbar(label='Cluster')
    plt.show()

def cluster_characterization(df, cluster_column):
    """
    Print statistical summaries for each cluster.

    Parameters:
    df (DataFrame): The DataFrame with cluster labels.
    cluster_column (str): The column name for cluster labels.
    """
    
    cluster_groups = df.groupby(cluster_column)
    for cluster, group in cluster_groups:
        print(f"Cluster {cluster} Characterization:")
        print(group.describe())
        print("\n")

if __name__ == '__main__':
    main()

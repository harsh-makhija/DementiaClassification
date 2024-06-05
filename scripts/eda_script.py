
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def basic_statistics(df):
    print("Basic Statistics:\n", df.describe())

def missing_values_analysis(df):
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values[missing_values > 0])

def categorical_analysis(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"Value Counts for {col}:\n", df[col].value_counts())
        sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts().values)
        plt.title(f'Distribution of {col}')
        plt.show()
        print()

def encode_cdr(df):
    # Encode CDR as numeric if it is categorical
    # Update this mapping based on actual values in CDR column
    cdr_mapping = {'none': 0, 'very mild': 0.5, 'mild': 1, 'moderate': 2}
    df['CDR_encoded'] = df['CDR'].map(cdr_mapping)
    return df

def correlation_matrix(df):
    # Encoding CDR column if it's not numeric
    df = encode_cdr(df)
    
    # Considering numerical and encoded CDR column for correlation matrix
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_df.corr(method='spearman')  # Using Spearman correlation
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def exploratory_data_analysis(file_path):
    df = load_data(file_path)

    # Basic statistics
    basic_statistics(df)

    # Missing values analysis
    missing_values_analysis(df)

    # Categorical data analysis
    categorical_analysis(df)

    # Correlation matrix
    correlation_matrix(df)

if __name__ == '__main__':
      
    exploratory_data_analysis(file_path)

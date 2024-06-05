import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def preprocess_data():
    """
    Preprocess the dataset by loading, cleaning, and merging multiple data files.
    This includes imputing missing values, correcting typos, and calculating some basic statistics.
    """

    # Paths to the data files
    data = pd.read_csv("/Users/harshmakhija/Desktop/CSC3432/data/visit-1.csv")
    df2 = pd.read_csv("/Users/harshmakhija/Desktop/CSC3432/data/visit-2.csv")
    df3 = pd.read_csv("/Users/harshmakhija/Desktop/CSC3432/data/visit-3.csv")
    df4 = pd.read_csv("/Users/harshmakhija/Desktop/CSC3432/data/visit-4.csv")
    df5 = pd.read_csv('/Users/harshmakhija/Desktop/CSC3432/data/visit-5.csv')

    # Load and concatenate the data from all files
    frames = [data, df2, df3, df4, df5]
    df = pd.concat(frames)

    # Convert 'ASF' values to numeric, handling comma as decimal separator
    df['ASF'] = df['ASF'].astype(str).str.replace(',', '.').astype(float)
    df['ASF'] = pd.to_numeric(df['ASF'], errors='coerce')

    # Define age ranges for analysis
    age_ranges = [(60, 69), (70, 79), (80, 89), (90, 99)]

    # Function to get mode for a range
    def get_mode_for_range(df, column, ranges):
        """
        Calculate the mode for a specified column within given ranges.

        Parameters:
        df (DataFrame): The DataFrame to analyze.
        column (str): The column to calculate the mode for.
        ranges (list of tuples): List of (min, max) pairs defining the ranges.

        Returns:
        dict: A dictionary with range as key and mode as value.
        """

        modes = {}
        for r in ranges:
            mode = df[(df[column] >= r[0]) & (df[column] <= r[1])]['SES'].mode()
            if not mode.empty:
                modes[r] = mode[0]
            else:
                modes[r] = np.nan
        return modes

    # Calculate mode SES for each age range
    mode_ses_by_age = get_mode_for_range(df, 'age', age_ranges)

    # Impute missing SES values based on age range
    for age_range in age_ranges:
        mode_ses = mode_ses_by_age[age_range]
        df.loc[(df['age'].between(age_range[0], age_range[1])) & (df['SES'].isna()), 'SES'] = mode_ses

    # Drop rows with missing ASF and eTIV
    df.dropna(subset=['ASF', 'eTIV', "MMSE"], inplace=True)

    # Correcting typos and inconsistencies in 'CDR' column
    cdr_mapping = {
        'very mild': 'very mild',
        'very miId': 'very mild',  # 'I' instead of 'l'
        'very midl': 'very mild',  # 'd' instead of 'l'
        'vry mild': 'very mild',   # missing 'e'
        'mild': 'mild',
        'midl': 'mild',            # typo
        'moderate': 'moderate',
        'none': 'none'
    }
    df['CDR'] = df['CDR'].map(cdr_mapping)

    # Save the processed data
    df.to_csv('/Users/harshmakhija/Desktop/CSC3432/data/processed.csv', index=False)
    print("Data preprocessing completed and saved to processed.csv")

    # Analyze categorical feature 'CDR'
    category_counts = df['CDR'].value_counts().reset_index()
    category_counts.columns = ['CDR Category', 'Count']

    # Analyze numerical feature 'age'
    age_summary = df['age'].agg(['mean', 'median', 'std']).reset_index()
    age_summary.columns = ['Statistic', 'Value']

    # Displaying the tables
    print("CDR Category Counts:")
    print(category_counts)
    print("\nAge Summary Statistics:")
    print(age_summary)

def main():
    preprocess_data()

if __name__ == "__main__":
    main()

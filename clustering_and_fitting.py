import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as ss



df = pd.read_csv('data.csv')


def plot_relational_plot(df):
    """
    Creates and saves a scatter plot of Hours Studied vs. Performance Index.
    The plot includes an improved legend for better readability.
    """
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({
        1: 'Yes', 0: 'No'
    }).astype(str)

    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=df, x='Hours Studied', y='Performance Index',
        hue='Extracurricular Activities'
    )
    handles, labels = scatter.get_legend_handles_labels()
    new_labels = ['No Extracurricular Activities (0)',
                  'Participates in Extracurricular Activities (1)']
    plt.legend(handles, new_labels, title='Extracurricular Activities')
    plt.title('Hours Studied vs Performance Index')
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    plt.grid(True)
    plt.savefig('relational_plot.png')


def plot_categorical_plot(df):
    """Creates and saves a bar plot of students participating in extracurricular activities."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Extracurricular Activities')
    plt.title('Extracurricular Activities Participation')
    plt.xlabel('Participation')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('categorical_plot.png')


def plot_statistical_plot(df):
    """
    Creates and saves a correlation heatmap for numerical features.
    The heatmap visually represents feature correlations.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')


def statistical_analysis(df, col):
    """
    Computes statistical properties of a column such as mean, stddev,
    skewness, kurtosis, and normality test results.
    """
    mean = float(df[col].mean())
    stddev = float(df[col].std())
    skew = float(df[col].skew())
    excess_kurtosis = float(df[col].kurtosis())

    # Normality Test - Use Shapiro-Wilk only if N ≤ 5000, otherwise use Kolmogorov-Smirnov
    if len(df[col]) <= 5000:
        stat, p_value = ss.shapiro(df[col])
        test_used = "Shapiro-Wilk Test"
    else:
        stat, p_value = ss.kstest(df[col], 'norm', args=(mean, stddev))
        test_used = "Kolmogorov-Smirnov Test"

    normality = "Normally distributed" if p_value > 0.05 else "Not normally distributed"

    # Print the results in a structured format
    print("\n")
    print('describe', '\n', df[col].describe(), '\n')
    print('correlation', '\n',  df.corr(), '\n')
    print('head', '\n',  df.head(), '\n')
    print('tail', '\n',  df.tail(), '\n')
    print(f"Stats for {col}:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std Dev: {stddev:.4f}")
    print(f"  Skewness: {skew:.6f}")
    print(f"  Excess Kurtosis: {excess_kurtosis:.5f}")
    print(f"  {test_used} p-value: {p_value:.6e}")
    print(f"  Normality: {normality}\n")

    return {
        'mean': mean,
        'stddev': stddev,
        'skew': skew,
        'excess_kurtosis': excess_kurtosis,
        'p_value': p_value,
        'normality': normality
    }



def perform_clustering(df, col1, col2):
    """
    Performs K-Means clustering on the dataset and visualizes the results.
    Generates both an elbow plot and a clustering plot.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): Column name for the first feature.
    col2 (str): Column name for the second feature.
    
    Returns:
    pd.DataFrame: The DataFrame with an additional 'Cluster' column.
    """
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # Elbow Plot
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertia, marker='o', linestyle='-', color='b', label='Inertia (Distortion Score)')
    plt.axvline(x=4, linestyle='--', color='red', label='Optimal Cluster (k=4)')
    plt.title('Elbow Plot for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.legend()
    plt.grid(True)
    plt.savefig('elbow_plot.png')
    plt.show()

    # Apply K-Means with optimal k (assumed 4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(data_scaled)

    # Define meaningful labels for clusters
    cluster_labels = {
        0: "Low Performance",
        1: "High Performance",
        2: "Moderate Performance",
        3: "Struggling"
    }

    # Cluster Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[col1], y=df[col2], hue=df['Cluster'], palette="deep", legend="full")

    # Update legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [cluster_labels[int(label)] for label in labels if label.isdigit()]
    plt.legend(handles, new_labels, title="Cluster Categories")

    plt.title("Clustering Plot of Study Hours vs Performance Index")
    plt.xlabel("Hours Studied")
    plt.ylabel("Performance Index")
    plt.grid(True)
    plt.savefig("clustering_plot.png")
    plt.show()

    return df

def perform_fitting(df, col1, col2):
    """
    Performs Linear and Polynomial Regression on the given dataset.
    Evaluates and plots both models to compare their performances.
    """
    X = df[[col1]].values
    y = df[col2].values
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_linear = lin_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='gray', label='Actual Data')
    plt.plot(X, y_pred_linear, color='red', label='Linear Fit')
    plt.plot(X, y_pred_poly, color='blue', linestyle='dashed',
             label='Polynomial Fit (Degree 2)')
    plt.title(f'Fitting: {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.grid(True)
    plt.savefig('fitting_plot.png')
    return r2_linear, r2_poly, rmse_linear, rmse_poly


def main():
    """
    Main function to execute data analysis pipeline.
    Reads data, performs statistical analysis, clustering, and regression fitting.
    """
    df = pd.read_csv('data.csv')
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    print(df.head())
    col = 'Performance Index'
    stats = statistical_analysis(df, col)
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    df = perform_clustering(df, 'Hours Studied', 'Performance Index')
    perform_fitting(df, 'Hours Studied', 'Performance Index')


if __name__ == '__main__':
    main()

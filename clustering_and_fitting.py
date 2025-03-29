import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error, r2_score
import scipy.stats as ss

df = pd.read_csv('data.csv')

def plot_relational_plot(df):
    """Creates and saves a scatter plot of Hours Studied vs. Performance Index."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Hours Studied', y='Performance Index', hue='Extracurricular Activities')
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
    """Creates and saves a correlation heatmap of numerical features."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')

def statistical_analysis(df, col):
    """Performs statistical analysis (mean, stddev, skewness, kurtosis, normality test)."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()

    stat, p_value = ss.shapiro(df[col])
    normality = "Normally distributed" if p_value > 0.05 else "Not normally distributed"

    return {
        'mean': mean,
        'stddev': stddev,
        'skew': skew,
        'excess_kurtosis': excess_kurtosis,
        'p_value': p_value,
        'normality': normality
    }

def perform_clustering(df, col1, col2):
    """Performs K-Means clustering on the dataset and visualizes it with improved readability."""
    data = df[[col1, col2]].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Using 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    df['Cluster'] = labels

    # Define custom colors and labels
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    cluster_labels = {
        0: "Low Study - Low Performance",
        1: "Moderate Study - Moderate Performance",
        2: "High Study - High Performance",
        3: "Low Study - High Performance"
    }

    # Scatter plot with improved colors and labels
    plt.figure(figsize=(8, 6))
    for cluster, color in cluster_colors.items():
        subset = df[df['Cluster'] == cluster]
        plt.scatter(subset[col1], subset[col2], c=color, label=cluster_labels[cluster], edgecolors='black', alpha=0.7)

    plt.title(f'Clustering of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend(title="Clusters", fontsize=10)
    plt.grid(True)
    plt.savefig('clustering_plot.png')

    return df



def perform_fitting(df, col1, col2):
    """Performs Linear and Polynomial Regression and evaluates their performance."""
    X = df[[col1]].values
    y = df[col2].values

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_linear = lin_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

    # Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

    print(f"Linear Regression: R² = {r2_linear:.3f}, RMSE = {rmse_linear:.3f}")
    print(f"Polynomial Regression (Degree 2): R² = {r2_poly:.3f}, RMSE = {rmse_poly:.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='gray', label='Actual Data')
    plt.plot(X, y_pred_linear, color='red', label='Linear Fit')
    plt.plot(X, y_pred_poly, color='blue', linestyle='dashed', label='Polynomial Fit (Degree 2)')
    plt.title(f'Fitting: {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.grid(True)
    plt.savefig('fitting_plot.png')

    return r2_linear, r2_poly, rmse_linear, rmse_poly

def main():
    df = pd.read_csv('data.csv')

    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    col = 'Performance Index'
    stats = statistical_analysis(df, col)
    print(f"Stats for {col}:")
    print(f"mean: {stats['mean']}")
    print(f"stddev: {stats['stddev']}")
    print(f"skew: {stats['skew']}")
    print(f"excess_kurtosis: {stats['excess_kurtosis']}")
    print(f"p_value: {stats['p_value']}")
    print(f"normality: {stats['normality']}")

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    df = perform_clustering(df, 'Hours Studied', 'Performance Index')

    perform_fitting(df, 'Hours Studied', 'Performance Index')

if __name__ == '__main__':
    main()

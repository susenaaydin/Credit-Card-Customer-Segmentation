# Credit-Card-Customer-Segmentation

A machine learning project that analyzes and segments credit card customers using **K-Means clustering**.

The project explores relationships between customer financial behaviors through data visualization and correlation analysis, determines an appropriate number of clusters using the **Elbow Method**, and reduces the dataset dimensions using **PCA** for visualization.

## Features

* 📊 Exploratory data analysis
* 🔎 Correlation analysis
* 📈 Customer behavior visualization
* 🔥 Correlation heatmap
* 📐 Feature standardization
* 📉 Elbow Method for determining the number of clusters
* 🤖 K-Means clustering
* 🧩 Customer segmentation
* 📊 PCA-based dimensionality reduction
* 🖥️ Data visualization with Matplotlib and Seaborn

## Dataset

The project uses the **Credit Card Customer Dataset** stored in:

```text
cc_general.csv
```

The dataset contains information about customer credit card usage and financial behavior.

The `CUST_ID` column is removed before clustering because it represents the customer identifier rather than a behavioral feature.

Missing values are handled using forward filling.

## Data Preprocessing

Before applying machine learning algorithms, the dataset is prepared through several steps:

1. Load the dataset using Pandas
2. Inspect the dataset
3. Fill missing values using forward fill
4. Remove the `CUST_ID` identifier
5. Standardize the numerical features using `StandardScaler`

Standardization ensures that features with different scales have a comparable influence on the clustering algorithm.

## Exploratory Data Analysis

Several scatter plots are used to investigate relationships between customer behavior variables.

The project analyzes relationships such as:

* Minimum Payments vs. Payments
* Credit Limit vs. Balance
* Credit Limit vs. Purchases
* Full Payment vs. Purchases
* Minimum Payments vs. Full Payments
* Cash Advance vs. Purchases

The plots use **TENURE** as a visual grouping variable.

## Correlation Analysis

A correlation matrix is calculated to examine relationships between numerical features.

A heatmap is generated using Seaborn to make correlations easier to interpret.

```python
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap=my_palette2)
```

## Determining the Number of Clusters

The **Elbow Method** is used to determine a suitable number of clusters.

The project evaluates cluster counts from 1 to 10 and calculates the **Within-Cluster Sum of Squares (WCSS)** for each value.

```python
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )
```

The resulting WCSS values are visualized to identify an appropriate cluster count.

## K-Means Clustering

The project uses the **K-Means** algorithm to segment customers into four clusters.

```python
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)

pred_y = kmeans.fit_predict(data_scaled)
```

Each customer is assigned to one of the four resulting clusters based on their financial behavior.

## PCA Dimensionality Reduction

Because the dataset contains multiple features, **Principal Component Analysis (PCA)** is used to reduce the dimensionality to two principal components.

```python
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
```

The resulting components are combined with the cluster labels to create a two-dimensional representation of the customer segments.

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Tkinter**
* **K-Means**
* **PCA**
* **StandardScaler**

## Project Workflow

```text
Credit Card Dataset
        ↓
Data Cleaning
        ↓
Feature Standardization
        ↓
Exploratory Data Analysis
        ↓
Correlation Analysis
        ↓
Elbow Method
        ↓
K-Means Clustering
        ↓
Customer Segmentation
        ↓
PCA Dimensionality Reduction
        ↓
Cluster Visualization
```

## Project Purpose

The purpose of this project is to demonstrate how machine learning can be used to identify groups of customers with similar financial behaviors.

The project provides practical experience with:

* Data preprocessing
* Exploratory data analysis
* Data visualization
* Correlation analysis
* Feature scaling
* Unsupervised machine learning
* K-Means clustering
* Dimensionality reduction with PCA

## Project Status

**Completed**

## Future Improvements

* Analyze and describe the characteristics of each customer cluster
* Add cluster visualization with PCA
* Compare different clustering algorithms
* Evaluate clustering quality using metrics such as Silhouette Score
* Create an interactive dashboard
* Add automated cluster analysis
* Generate customer segment reports

## Author

**Sude Sena Aydın**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('train.csv')

# Select features and fill missing values
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
X = df[features].copy()
X.fillna(X.median(), inplace=True)

# Scale the data for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Elbow Method to find the optimal number of clusters (k)
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# 2. Apply K-Means clustering (using k=3 based on the elbow method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 3. 2D PCA Projection for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.subplot(1, 2, 2)
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.7)
plt.title('2D PCA Projection of House Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# 4. Cluster Analysis
print("\n CLUSTER PROFILES ")
cluster_analysis = df.groupby('Cluster')[features + ['SalePrice']].mean().round(2)
print(cluster_analysis)

print("\n TOP NEIGHBORHOODS PER CLUSTER")
for i in range(3):
    print(f"\nCluster {i} Top Neighborhoods:")
    print(df[df['Cluster'] == i]['Neighborhood'].value_counts().head(3))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Make pandas output look wider and prevent column wrapping
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

print("\n" + "="*60)
print(" FIRST 5 ROWS OF DATA")
print("="*60)
print(df.head())

print("\n" + "="*60)
print(" DATA INFO")
print("="*60)
df.info()

print("\n" + "="*60)
print(" DESCRIPTIVE STATISTICS (Rounded)")
print("="*60)
print(df.describe().round(2))
print("="*60 + "\n")


# 1. 2D Clustering: Annual Income vs Spending Score
X_2d = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans_2d = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans_2d.fit_predict(X_2d)

plt.figure(figsize=(10,5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')

centroids_2d = kmeans_2d.cluster_centers_
plt.scatter(centroids_2d[:,0], centroids_2d[:,1], marker='X', s=100, c='red')

plt.title("Customer segmentation using k-means (Income vs Spending)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show() 


# 2. 3D Clustering: Income, Spending, Age
X_3d = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

kmeans_3d = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_3d'] = kmeans_3d.fit_predict(X_3d)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df['Age'], 
           c=df['Cluster_3d'], cmap='viridis', s=60)

centroids_3d = kmeans_3d.cluster_centers_
ax.scatter(centroids_3d[:,0], centroids_3d[:,1], centroids_3d[:,2], 
           marker='X', s=100, c='red', label='Centroids')

ax.set_title("3D Customer segmentation using k-means")
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
plt.legend()
plt.show()

# 3. 2D Clustering: Spending Score vs Age
X_spend_age = df[['Spending Score (1-100)','Age']]

kmeans_spend_age = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_spend_age'] = kmeans_spend_age.fit_predict(X_spend_age)

plt.figure(figsize=(12,8))
plt.scatter(df['Spending Score (1-100)'], df['Age'], c=df['Cluster_spend_age'], cmap='viridis')

centroids_spend_age = kmeans_spend_age.cluster_centers_
plt.scatter(centroids_spend_age[:,0], centroids_spend_age[:,1], marker='X', s=100, c='red')

plt.title("Customer segmentation (Spending Score vs Age)")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Age")
plt.show()


# 4. 2D Clustering: Annual Income vs Age
X_inc_age = df[['Annual Income (k$)','Age']]

kmeans_inc_age = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_inc_age'] = kmeans_inc_age.fit_predict(X_inc_age)

plt.figure(figsize=(12,8))
plt.scatter(df['Annual Income (k$)'], df['Age'], c=df['Cluster_inc_age'], cmap='viridis')

centroids_inc_age = kmeans_inc_age.cluster_centers_
plt.scatter(centroids_inc_age[:,0], centroids_inc_age[:,1], marker='X', s=100, c='red')

plt.title("Customer segmentation (Annual Income vs Age)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Age")
plt.show()

# 5. 4D Clustering (Income, Spending, Age, Gender) using Matplotlib
# Encode Gender to numbers
le = LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Genre'])

X_4d = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Gender_encoded']]

kmeans_4d = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster4d'] = kmeans_4d.fit_predict(X_4d)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Loop through the genders to assign different shapes (circle for one, triangle for the other)
markers = ['o', '^'] 
for gender_val, marker in zip(df['Gender_encoded'].unique(), markers):
    subset = df[df['Gender_encoded'] == gender_val]
    ax.scatter(subset['Annual Income (k$)'], 
               subset['Spending Score (1-100)'], 
               subset['Age'], 
               c=subset['Cluster4d'], 
               cmap='viridis', 
               s=60, 
               marker=marker,
               label=f"Gender {gender_val}")

ax.set_title("4D segmentation (Color = Cluster, Shape = Gender)")
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
plt.legend()
plt.show()
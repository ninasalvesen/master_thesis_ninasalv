import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)
df1['XY'] = np.sqrt(df1['X']**2 + df1['Y']**2)
df1 = df1.drop(columns=['Datetime', 'Lat', 'X', 'Y', 'Lon', 'Data set', 'uniq.log', 'Farm', 'minutes'])

#print(df1)
#print(df1.describe())


def Standardize(df, columns):  # standardizes the columns listed in the variable columns
    scaler = preprocessing.StandardScaler()
    for i in range(len(columns)):
        column = np.array(df[columns[i]]).reshape(-1, 1)
        scaled = pd.DataFrame(scaler.fit_transform(column), columns=[columns[i]])
        df.drop(columns=[columns[i]], inplace=True)
        df = pd.concat([scaled, df], axis=1)
    return df


def Normalize(df, columns):  # normalizes the columns listed in the variable columns
    scaler = preprocessing.MinMaxScaler()
    for i in range(len(columns)):
        column = np.array(df[columns[i]]).reshape(-1, 1)
        scaled = pd.DataFrame(scaler.fit_transform(column), columns=[columns[i]])
        df.drop(columns=[columns[i]], inplace=True)
        df = pd.concat([scaled, df], axis=1)
    return df


df1 = Standardize(df1, ['Haversine', 'XY'])
df1 = Normalize(df1, ['Haversine', 'XY'])
print(df1)
#print(df1.describe())



# Elbow method of finding number of clusters for optimal kmeans
sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1)
    df1['clusters'] = kmeans.labels_  # .labels_ er det samme som .predict når man putter inn samme datasett i predict
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
plt.figure(figsize=(16, 8))
plt.plot(list(sse.keys()), list(sse.values()))  # gjør dette til ditt eget så det ikke kan plagieres?
plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35, pad=15)
plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
plt.ylabel('SSE', fontsize=25, labelpad=15)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/elbow_kmeans_tingvoll.png", dpi=500)
plt.show()
# optimal k=6

kmeans = KMeans(n_clusters=6)
# forklaring på hver iterasjon og toleranse + center shift: verbose=2 i Kmeans
kmeans.fit(df1)
centroids = kmeans.cluster_centers_
print(centroids)
clusters = kmeans.labels_  # samme som kmeans.predict(df1)
df1['cluster'] = clusters
print(df1)
print('inertia:', kmeans.inertia_)
print('iterations:', kmeans.n_iter_)
print('n features:', kmeans.n_features_in_)
print('features:', kmeans.feature_names_in_)
# print('Silhouette score:', metrics.silhouette_score(df1, clusters))  
# kjører veldig lenge O(n**2), ikke bruk på så mye data

fig1 = px.scatter_3d(df1, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
fig1.update_layout(title='3D cluster rep. Tingvoll')
fig1.show()



# try dimensionality reduction with PCA:

# first test what dimension to use
pca = PCA()
pca.fit(df1)
print(pca.explained_variance_ratio_)
plt.figure(figsize=(16, 8))
plt.plot(range(1, 5), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained variance by number of features', fontsize=35, pad=15)
plt.xlabel('Number of features', fontsize=25, labelpad=15)
plt.ylabel('Cumulative explained variance', fontsize=25, labelpad=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/PCA_feature_variance_tingvoll.png", dpi=500)
plt.show()
# bør holde ca. 0.8 av variansen, her bør dimensjonen bli 2

pca = PCA(n_components=2)
pca.fit(df1)
df1_reduced = pd.DataFrame(pca.transform(df1), columns=['PC1', 'PC2'])

print(df1_reduced)
sse = {}
for k in range(1, 20):
    print(k)
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1_reduced)
    df1_reduced['clusters'] = kmeans.labels_  # .labels_ er det samme som .predict når man putter inn samme datasett i predict
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
plt.figure(figsize=(16, 8))
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35, pad=15)
plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
plt.ylabel('SSE', fontsize=25, labelpad=15)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/elbow_kmeans_PCA_tingvoll.png", dpi=500)
plt.show()
# optimal k=4

kmeans = KMeans(n_clusters=4)
kmeans.fit(df1_reduced)
centroids = kmeans.cluster_centers_
print(centroids)
print(centroids[0])
clusters = kmeans.labels_
df1_reduced['cluster'] = clusters
print(df1_reduced)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=df1_reduced['PC1'], y=df1_reduced['PC2'], hue=df1_reduced['cluster'], palette=['g', 'r', 'c', 'm'])
plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x')
plt.title('Kmeans clusters with PCA dimension reduction', fontsize=35, pad=15)
plt.xlabel('PC1', fontsize=25, labelpad=15)
plt.ylabel('PC2', fontsize=25, labelpad=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans_PCA_tingvoll.png", dpi=500)
plt.show()

#df1_reduced['cluster'] = clusters
#print(df1_reduced)

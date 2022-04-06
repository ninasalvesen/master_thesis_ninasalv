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

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V7 sinetime uten 2018.csv',
                 delimiter=';', low_memory=False)

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%d/%m/%Y %H:%M')


#df1.dropna(subset=['Datetime'], inplace=True)
#df1.reset_index(inplace=True, drop=True)
#df1['XY'] = np.sqrt(df1['X']**2 + df1['Y']**2)
#df1 = df1.drop(columns=['Datetime', 'Lat', 'X', 'Y', 'Lon', 'Data set', 'uniq.log', 'Farm', 'minutes'])

df2.dropna(subset=['Datetime'], inplace=True)
df2.reset_index(inplace=True, drop=True)
df2 = df2.drop(columns=['Datetime', 'Lat', 'Lon', 'Data set', 'uniq.log', 'minutes'])


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


def ElbowMethod(df, fig=False):
    sse = []
    for k in range(1, 21):
        print(k)
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
        df['clusters'] = kmeans.labels_  # .labels_ er det samme som .predict når man putter inn samme datasett i predict
        sse.append(kmeans.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, 21), sse)  # gjør dette til ditt eget så det ikke kan plagieres?
    plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35, pad=15)
    plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
    plt.ylabel('SSE', fontsize=25, labelpad=15)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans/fosen_elbow_kmeans_tessst.png", dpi=500)
    plt.show()


def Kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    # forklaring på hver iterasjon og toleranse + center shift: verbose=2 i KMeans

    kmeans.fit(df)
    #centroids = kmeans.cluster_centers_
    #print(centroids)
    clusters = kmeans.labels_  # samme som kmeans.predict(df1)
    df['cluster'] = clusters
    print('inertia:', kmeans.inertia_)
    print('iterations:', kmeans.n_iter_)
    print('n features:', kmeans.n_features_in_)
    print('features:', kmeans.feature_names_in_)
    # print('Silhouette score:', metrics.silhouette_score(df1, clusters))
    # kjører veldig lenge O(n**2), ikke bruk på så mye data

    fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', color='cluster', opacity=1)
    #fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
    fig.update_layout(title='3D cluster rep. Tingvoll')
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    fig.show()


def FindDimension(df, n_features, fig=False):
    pca = PCA()
    pca.fit(df)
    #print(pca.explained_variance_ratio_)
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, n_features+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained variance by number of features', fontsize=35, pad=15)
    plt.xlabel('Number of features', fontsize=25, labelpad=15)
    plt.ylabel('Cumulative explained variance', fontsize=25, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans/tingvoll_PCA_feature_variance.png", dpi=500)
    plt.show()
    # bør holde ca. 0.8 av variansen


def ElbowPCA(df, dimension, fig=False):
    pca = PCA(n_components=dimension)
    pca.fit(df)
    df_reduced = pd.DataFrame(pca.transform(df), columns=['PC1', 'PC2'])

    print(df_reduced)
    sse = []
    for k in range(1, 21):
        print(k)
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_reduced)
        df_reduced['clusters'] = kmeans.labels_  # .labels_ er det samme som .predict når man putter inn samme datasett i predict
        sse[k].append(kmeans.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, 21), sse)
    plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35, pad=15)
    plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
    plt.ylabel('SSE', fontsize=25, labelpad=15)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans/tingvoll_elbow_kmeans_PCA.png", dpi=500)
    plt.show()
    return df_reduced


def KmeansPCA(df_reduced, n_clusters, fig=False):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_reduced)
    centroids = kmeans.cluster_centers_
    #print(centroids)
    #print(centroids[0])
    clusters = kmeans.labels_
    df_reduced['cluster'] = clusters

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df_reduced['PC1'], y=df_reduced['PC2'], hue=df_reduced['cluster'], palette=['g', 'r', 'c', 'm'])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='x')
    plt.title('Kmeans clusters with PCA dimension reduction', fontsize=35, pad=15)
    plt.xlabel('PC1', fontsize=25, labelpad=15)
    plt.ylabel('PC2', fontsize=25, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans/tingvoll_kmeans_PCA.png", dpi=500)
    plt.show()



# TINGVOLL:
#df1 = Standardize(df1, ['Haversine', 'XY'])
#df1 = Normalize(df1, ['Haversine', 'XY'])
#print(df1)
#print(df1.describe())

#ElbowMethod(df1, False)  # k=6
#Kmeans(df1, 6)
#FindDimension(df1, 4, False)  # n=2
#df1_reduced = ElbowPCA(df1, 2, False)  # k=4
#KmeansPCA(df1_reduced, 4, False)


# FOSEN:
#df2 = Standardize(df2, ['Haversine'])
#df2 = Normalize(df2, ['Haversine'])  # skal jeg ha med både standard og normal her?
#print(df2.describe())

#ElbowMethod(df2, False)  # k=4 for standardize+normalize, k=6 for kun standardize
#Kmeans(df2, 4)







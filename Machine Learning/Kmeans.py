import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')
#print(df1.columns)

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)

df1 = df1.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log',
                        'besetning', 'race'])
#print(df1.columns)
#'Temp', 'angle', 'Altitude', 'n_lambs', 'age'


def Standardize(df, columns):  # standardizes the columns listed in the variable columns
    scaler = preprocessing.StandardScaler()
    for i in range(len(columns)):
        column = np.array(df[columns[i]]).reshape(-1, 1)
        scaled = pd.DataFrame(scaler.fit_transform(column), columns=[columns[i]])
        df.drop(columns=[columns[i]], inplace=True)
        df = pd.concat([scaled, df], axis=1)
    return df


def Normalize(df, columns, a, b):  # normalizes the columns listed in the variable columns
    scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
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
    plt.plot(range(1, 21), sse)
    plt.title('Elbow method for Kmeans++, dynamic features and n_lambs for all data', fontsize=35, pad=15)
    plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
    plt.ylabel('SSE', fontsize=25, labelpad=15)
    plt.grid(True)
    xlabels = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    plt.xticks(fontsize=20, ticks=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20], labels=xlabels)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/kmeans/kmeans_elbow_ageno.png", dpi=500)
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

    #fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Velocity', color='cluster', opacity=1)
    #fig.update_traces(marker=dict(size=5, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    #fig.update_layout(title='Kmeans++ for lighter sheep breeds')

    polar = df.groupby('cluster').mean().reset_index()
    print(polar['Altitude'].describe(), polar['angle'].describe(), polar['Temp'].describe(), polar['Velocity'].describe())
    print(polar['age'].describe(), polar['n_lambs'].describe(), polar['sin_time'].describe(), polar['cos_time'].describe())
    polar1 = df.groupby('cluster').std().reset_index()
    polar = pd.melt(polar, id_vars=['cluster'])
    polar1 = pd.melt(polar1, id_vars=['cluster'])
    fig = px.line_polar(polar, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)
    fig1 = px.line_polar(polar1, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)

    fig.show()
    fig1.show()


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


#df1 = Standardize(df1, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'])
#df1 = Normalize(df1, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'], -1, 1)
#ElbowMethod(df1, True)  # k=4
#Kmeans(df1, 4)


#print(len(df1['uniq.log'].unique()))


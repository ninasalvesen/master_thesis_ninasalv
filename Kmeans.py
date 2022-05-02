import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
"""
df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V9 med temp.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V9 info_features med temp.csv',
                 delimiter=';', low_memory=False)

df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')
"""
#dftot = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total.csv', delimiter=';', low_memory=False)
#dftot['Datetime'] = pd.to_datetime(dftot['Datetime'], format='%Y-%m-%d %H:%M:%S')
#dfnew = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase.csv', delimiter=';', low_memory=False)
#dfold = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase.csv', delimiter=';', low_memory=False)

#df_tidlig = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total tidlig sesong.csv', delimiter=';', low_memory=False)
#df_sen = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total sen sesong.csv', delimiter=';', low_memory=False)
#df_tot_gammel = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase.csv', delimiter=';', low_memory=False)
#df_tot_ny = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase.csv', delimiter=';', low_memory=False)

"""
df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)
#df1['XY'] = np.sqrt(df1['X']**2 + df1['Y']**2)
df1 = df1.drop(columns=['Lat', 'X', 'Y', 'Lon', 'Farm', 'minutes'])

df2.dropna(subset=['Datetime'], inplace=True)
df2.reset_index(inplace=True, drop=True)
df2 = df2.drop(columns=['Lat', 'Lon', 'minutes'])

df_tot = pd.concat([df1, df2])
df_tot.reset_index(inplace=True, drop=True)
print(df_tot)
#df_tot.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total.csv', index=False, sep=';')
"""

#df3.dropna(subset=['race'], inplace=True)
#df3.reset_index(inplace=True, drop=True)
#df3.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total.csv', index=False, sep=';')

#df_old = df3[df3['race'] != 'NKS']
#df_new = df3[df3['race'] == 'NKS']
#df_old.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase.csv', index=False, sep=';')
#df_new.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase.csv', index=False, sep=';')


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
    plt.plot(range(1, 21), sse)  # gjør dette til ditt eget så det ikke kan plagieres?
    plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35, pad=15)
    plt.xlabel('Number of clusters', fontsize=25, labelpad=15)
    plt.ylabel('SSE', fontsize=25, labelpad=15)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if fig:
        plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/kmeans/tingvoll_elbow_kmeans_tessst.png", dpi=500)
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
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    #fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
    fig.update_layout(title='3D cluster rep. Tingvoll')
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
#df1 = Normalize(df1, ['Haversine', 'XY'], -1, 1)

#ElbowMethod(df1, False)  # k=6, k=4
#Kmeans(df1, 4)


# FOSEN:
#df2 = Standardize(df2, ['Haversine'])
#df2 = Normalize(df2, ['Haversine'])  # skal jeg ha med både standard og normal her?

#ElbowMethod(df2, False)  # k=4 for standardize+normalize, k=6 for kun standardize
#Kmeans(df2, 4)


# Total
#print(len(dftot['uniq.log'].unique()))
#dftot = dftot.drop(columns=['uniq.log', 'race', 'besetning', 'age', 'n_lambs', 'Temp'])
#print(dftot.columns)
#dftot = Standardize(dftot, ['Haversine'])
#dftot = Normalize(dftot, ['Haversine'], -1, 1)

#ElbowMethod(dftot, False) # k=4
#Kmeans(dftot, 4)


"""
for i in range(len(dftot)):
    dftot.at[i, 'month'] = dftot.at[i, 'Datetime'].month

df_early = dftot[dftot['month'] < 7]
df_late = dftot[dftot['month'] >= 7]
#df_early.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total tidlig sesong.csv', index=False, sep=';')
#df_late.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Total sen sesong.csv', index=False, sep=';')

df_early_old = df_early[df_early['race'] != 'NKS']
df_early_new = df_early[df_early['race'] == 'NKS']
#df_early_old.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase tidlig sesong.csv', index=False, sep=';')
#df_early_new.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase tidlig sesong.csv', index=False, sep=';')

df_late_old = df_late[df_late['race'] != 'NKS']
df_late_new = df_late[df_late['race'] == 'NKS']
#df_late_old.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase sen sesong.csv', index=False, sep=';')
#df_late_new.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase sen sesong.csv', index=False, sep=';')


# Early season
#print(len(dftot['uniq.log'].unique()))
df_tidlig = df_tidlig.drop(columns=['Datetime', 'uniq.log', 'month', 'race', 'besetning', 'age', 'n_lambs', 'Temp'])
print(df_tidlig.columns)
df_tidlig = Standardize(df_tidlig, ['Haversine'])
df_tidlig = Normalize(df_tidlig, ['Haversine'], -1, 1)

#ElbowMethod(df_tidlig, False) # k=4
#Kmeans(df_tidlig, 4)

# Late season
#print(len(dftot['uniq.log'].unique()))
df_sen = df_sen.drop(columns=['Datetime', 'uniq.log', 'month', 'race', 'besetning', 'age', 'n_lambs', 'Temp'])
print(df_sen.columns)
df_sen = Standardize(df_sen, ['Haversine'])
df_sen = Normalize(df_sen, ['Haversine'], -1, 1)

#ElbowMethod(df_sen, False) # k=4
Kmeans(df_sen, 4)


# Tot old
#print(len(dftot['uniq.log'].unique()))
df_tot_gammel = df_tot_gammel.drop(columns=['Datetime', 'uniq.log', 'race', 'besetning', 'age', 'n_lambs', 'Temp'])
print(df_tot_gammel.columns)
df_tot_gammel = Standardize(df_tot_gammel, ['Haversine'])
df_tot_gammel = Normalize(df_tot_gammel, ['Haversine'], -1, 1)

#ElbowMethod(df_tot_gammel, False) # k=4
Kmeans(df_tot_gammel, 4)


# Tot ny
#print(len(dftot['uniq.log'].unique()))
df_tot_ny = df_tot_ny.drop(columns=['Datetime', 'uniq.log', 'race', 'besetning', 'age', 'n_lambs', 'Temp'])
print(df_tot_ny.columns)
df_tot_ny = Standardize(df_tot_ny, ['Haversine'])
df_tot_ny = Normalize(df_tot_ny, ['Haversine'], -1, 1)

#ElbowMethod(df_tot_ny, False) # k=4
Kmeans(df_tot_ny, 4)
"""

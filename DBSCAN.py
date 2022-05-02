import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from Kmeans import Standardize, Normalize
import seaborn as sns
"""
df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V9 med temp.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


# removing empty rows at the beginning of each set
df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)

# chose which features to include
#df1['XY'] = np.sqrt(df1['X']**2 + df1['Y']**2)
df1 = df1.drop(columns=['Datetime', 'Lat', 'X', 'Y', 'Lon', 'Data set', 'uniq.log', 'Farm', 'minutes',
                        'race', 'besetning'])
print(df1)
"""
df_tot_gammel = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Gammel_rase.csv', delimiter=';', low_memory=False)
df_tot_ny = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Ny_rase.csv', delimiter=';', low_memory=False)
df_tot_ny = df_tot_ny.drop(columns=['Datetime', 'uniq.log', 'race', 'besetning'])
df_tot_gammel = df_tot_gammel.drop(columns=['Datetime', 'uniq.log', 'race', 'besetning'])

#print(df_tot_gammel['Haversine'].describe())
#print(df_tot_ny['Haversine'].describe())


def dbscan(df, epsilon, min, fig=False):
    db = DBSCAN(eps=epsilon, min_samples=min)
    db.fit(df)
    clusters = db.labels_
    df['cluster'] = clusters

    print('n features:', db.n_features_in_)
    print('features:', db.feature_names_in_)

    no_clusters = len(np.unique(clusters))
    no_noise = np.sum(np.array(clusters) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    if fig:
        df['Temp'] = df['Temp'] + 1
        fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', size='Temp', color='cluster', opacity=1)
        #fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        # fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
        fig.update_layout(title='3D cluster rep. Tingvoll DBSCAN')
        fig.show()



# Tingvoll
#pd.plotting.scatter_matrix(df1)
#plt.show()
#df1 = Standardize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'])
#df1 = Normalize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'], -1, 1)
#print(df1.describe())
#print(df1)


# find eps by using elbow method by KNN
def epsPlot(df, n, r, fig=False):
    knn = NearestNeighbors(n_neighbors=n, radius=r)
    knn.fit(df)

    dist, ind = knn.kneighbors(df)

    dist = np.sort(dist, axis=0)
    dist = dist[:, n - 1]

    if fig:
        plt.figure(figsize=(8, 8))
        plt.plot(dist)
        plt.xlabel('Points in the dataset', fontsize=12)
        plt.ylabel('Sorted {}-nearest neighbor distance'.format(n), fontsize=12)
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()


# Tingvoll
#k = 2 * df1.shape[-1] - 1
#epsPlot(df1, k, 1)
#dbscan(df1, 0.2, 12, True)
#df_noise = df1[df1['cluster'] == -1]
#df_noise = df_noise.drop(columns=['cluster'])
#print(df_noise.describe())
#corr = df1.corr()
#sns.heatmap(corr)

# Gammel rase
#print(df_tot_gammel.columns)
df_tot_gammel = Standardize(df_tot_gammel, ['Haversine', 'age', 'n_lambs', 'Temp'])
df_tot_gammel = Normalize(df_tot_gammel, ['Haversine', 'age', 'n_lambs', 'Temp'], -1, 1)

#k = 2 * df_tot_gammel.shape[-1] - 1
#epsPlot(df_tot_gammel, k, 1, True)
dbscan(df_tot_gammel, 0.135, 11, False)
df_noise = df_tot_gammel[df_tot_gammel['cluster'] == -1]
df_noise = df_noise.drop(columns=['cluster', 'age', 'n_lambs'])
print(df_noise.describe())
print(len(df_noise[df_noise['Haversine'] < -0.894457]))

plt.figure()
plt.hist(df_noise['Haversine'], bins=50)
plt.axvline(x=-0.894457)
plt.show()
#corr = df1.corr()
#sns.heatmap(corr)
"""
# Ny rase
df_tot_ny = Standardize(df_tot_ny, ['Haversine', 'age', 'n_lambs', 'Temp'])
print(df_tot_ny['Haversine'].describe())
df_tot_ny = Normalize(df_tot_ny, ['Haversine', 'age', 'n_lambs', 'Temp'], -1, 1)

k = 2 * df_tot_ny.shape[-1] - 1
#epsPlot(df_tot_ny, k, 1, True)
dbscan(df_tot_ny, 0.135, 11)
df_noise = df_tot_ny[df_tot_ny['cluster'] == -1]
df_noise = df_noise.drop(columns=['cluster', 'age', 'n_lambs'])
print(df_noise.describe())
#corr = df1.corr()
#sns.heatmap(corr)
"""




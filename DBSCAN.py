import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from Kmeans import Standardize, Normalize
import seaborn as sns

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
df1 = Standardize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'])
df1 = Normalize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'], -1, 1)
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


#k = 2 * df1.shape[-1] - 1
#epsPlot(df1, k, 1)
#dbscan(df1, 0.2, 12, True)
#df_noise = df1[df1['cluster'] == -1]
#df_noise = df_noise.drop(columns=['cluster'])
#print(df_noise.describe())
#corr = df1.corr()
#sns.heatmap(corr)




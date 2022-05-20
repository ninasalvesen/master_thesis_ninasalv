import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from Kmeans import Standardize, Normalize
import seaborn as sns

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')


# removing empty rows at the beginning of each set
df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)
df_old = df1[df1['race'] != 'NKS']
df_new = df1[df1['race'] == 'NKS']

# chose which features to include
df1 = df1.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])

df_old = df_old.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])

df_new = df_new.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])

#'Temp', 'angle', 'Altitude', 'n_lambs', 'age'
#print(df1)

#df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise.csv', delimiter=';', low_memory=False)
#print(df2)
#df2 = df2.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])
#print(df2.describe())


def dbscan(df, epsilon, min, fig=False):
    db = DBSCAN(eps=epsilon, min_samples=min, n_jobs=-1)
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
        #df['Temp'] = df['Temp'] + 1
        #fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Velocity', color='cluster', opacity=1)
        #fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

        polar = df.groupby('cluster').mean().reset_index()

        print(polar['Altitude'].describe(), polar['angle'].describe(), polar['Temp'].describe(), polar['Velocity'].describe())
        print(polar['age'].describe(), polar['n_lambs'].describe(), polar['sin_time'].describe(), polar['cos_time'].describe())

        polar1 = df.groupby('cluster').std().reset_index()
        polar = pd.melt(polar, id_vars=['cluster'])
        polar1 = pd.melt(polar1, id_vars=['cluster'])
        fig = px.line_polar(polar, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)
        fig1 = px.line_polar(polar1, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)

        #fig.show()
        #fig1.show()


# find eps by using elbow method by KNN
def epsPlot(df, n, r, fig=False):
    knn = NearestNeighbors(n_neighbors=n, radius=r)
    knn.fit(df)

    dist, ind = knn.kneighbors(df)

    dist = np.sort(dist, axis=0)
    dist = dist[:, n - 1]
    if fig:
        plt.figure(figsize=(12, 8))
        plt.plot(dist)
        plt.title('Epsilon plot, all data', fontsize=35, pad=15)
        plt.xlabel('Points in the dataset', fontsize=25, labelpad=15)
        plt.ylabel('Sorted {}-nearest neighbor distance'.format(n), fontsize=25, labelpad=15)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        #plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/dbscan/epsplot_all.png", dpi=500)
        plt.show()


#df1 = Standardize(df1, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'])
#df1 = Normalize(df1, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'], -1, 1)

#k = 2 * df1.shape[-1] - 1
#epsPlot(df1, k, 1, True)
#dbscan(df1, 0.37, 16, True)
#df_noise = df1[df1['cluster'] == -1]
#df_noise = df_noise.drop(columns=['cluster'])
#print(df_noise.describe())
#df_noise.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise.csv', index=False, sep=';')

"""
print(len(df_noise[df_noise['Haversine'] < -0.894457]))
plt.figure()
plt.hist(df_noise['Haversine'], bins=50)
plt.axvline(x=-0.894457)
plt.show()
"""



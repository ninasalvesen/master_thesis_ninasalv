import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from Kmeans import Standardize, Normalize

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

def dbscan(df, epsilon, min):
    db = DBSCAN(eps=epsilon, min_samples=min).fit(df1)
    clusters = db.labels_
    df['clusters'] = clusters

    print('n features:', db.n_features_in_)
    print('features:', db.feature_names_in_)

    no_clusters = len(np.unique(clusters))
    no_noise = np.sum(np.array(clusters) == -1, axis=0)

    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)

    fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', color='clusters', opacity=1)
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    # fig = px.scatter_3d(df, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
    fig.update_layout(title='3D cluster rep. Tingvoll DBSCAN')
    fig.show()



# Tingvoll
df1 = Standardize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'])
df1 = Normalize(df1, ['Haversine', 'age', 'n_lambs', 'Temp'], -1, 1)
#print(df1.describe())
print(df1)
dbscan(df1, 0.2, 12)

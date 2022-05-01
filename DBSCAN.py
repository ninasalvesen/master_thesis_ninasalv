import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V9 med temp.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df1)

"""

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)
df1['XY'] = np.sqrt(df1['X']**2 + df1['Y']**2)
df1 = df1.drop(columns=['Datetime', 'Lat', 'X', 'Y', 'Lon', 'Data set', 'uniq.log', 'Farm', 'minutes'])

epsilon = 1
min_samples = 15

db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(df1)
labels = db.labels_

no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)
"""

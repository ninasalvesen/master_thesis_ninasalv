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
"""
# initiating new sets from data splits on race, season and daytime

# race
df_old = df1[df1['race'] != 'NKS']
df_new = df1[df1['race'] == 'NKS']
df_old.reset_index(inplace=True, drop=True)
df_new.reset_index(inplace=True, drop=True)

# daytime
for i in range(len(df1)):
    df1.at[i, 'daytime'] = int(df1.at[i, 'Datetime'].hour)
    if (df1.at[i, 'daytime'] >= 4) and (df1.at[i, 'daytime'] <= 21):
        df1.at[i, 'type'] = 'day'
    else:
        df1.at[i, 'type'] = 'night'
df_day = df1[df1['type'] == 'day']
df_night = df1[df1['type'] == 'night']
df_day.reset_index(inplace=True, drop=True)
df_night.reset_index(inplace=True, drop=True)

# season
for i in range(len(df1)):
    df1.at[i, 'month'] = int(df1.at[i, 'Datetime'].month)
df_early = df1[df1['month'] < 7]
df_late = df1[df1['month'] >= 7]
df_early.reset_index(inplace=True, drop=True)
df_late.reset_index(inplace=True, drop=True)
"""

# chose which features to include
# 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'
df1 = df1.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])
"""
df_old = df_old.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])
df_new = df_new.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race'])

df_day = df_day.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race', 'daytime', 'type'])
df_night = df_night.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race', 'daytime', 'type'])

df_early = df_early.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race', 'month'])
df_late = df_late.drop(columns=['Lat', 'Lon', 'Datetime', 'Data set', 'uniq.log', 'besetning', 'race', 'month'])

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_new.csv', delimiter=';', low_memory=False)
df2 = df2.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])
"
df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_old.csv', delimiter=';', low_memory=False)
df3 = df3.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df4 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_early.csv', delimiter=';', low_memory=False)
df4 = df4.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df5 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_late.csv', delimiter=';', low_memory=False)
df5 = df5.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df6 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_day.csv', delimiter=';', low_memory=False)
df6 = df6.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])

df7 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_night.csv', delimiter=';', low_memory=False)
df7 = df7.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])
"""
df8 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise.csv', delimiter=';', low_memory=False)
df8 = df8.drop(columns=['sin_time', 'cos_time', 'age', 'n_lambs'])
"""
# print statistical information
print('new/heavy:')
print(df2.describe())
print('---------------------------------------------------------')
print('old/light:')
print(df3.describe())
print('---------------------------------------------------------')
print('early season:')
print(df4.describe())
print('---------------------------------------------------------')
print('late season:')
print(df5.describe())
print('---------------------------------------------------------')
print('day:')
print(df6.describe())
print('---------------------------------------------------------')
print('night:')
print(df7.describe())
print('---------------------------------------------------------')
"""


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
        #print(polar['Altitude'].describe(), polar['angle'].describe(), polar['Temp'].describe(), polar['Velocity'].describe())
        #print(polar['age'].describe(), polar['n_lambs'].describe(), polar['sin_time'].describe(), polar['cos_time'].describe())
        polar1 = df.groupby('cluster').std().reset_index()
        polar = pd.melt(polar, id_vars=['cluster'])
        polar1 = pd.melt(polar1, id_vars=['cluster'])

        fig = px.line_polar(polar, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)
        fig1 = px.line_polar(polar1, r='value', theta='variable', color='cluster', line_close=True, height=800, width=1400)

        fig.show()
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
        plt.title('Epsilon plot, nighttime', fontsize=35, pad=15)
        plt.xlabel('Points in the dataset', fontsize=25, labelpad=15)
        plt.ylabel('Sorted {}-nearest neighbor distance'.format(n), fontsize=25, labelpad=15)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.05, 1.9)
        plt.tight_layout()
        #plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/dbscan/dbscan_eps_night.png", dpi=500)
        plt.show()


def runner(df):
    #print(df['Velocity'].describe())
    df = Standardize(df, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'])
    #print(df['Velocity'].describe())
    df = Normalize(df, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'], -1, 1)
    #k = 2 * df.shape[-1] - 1
    #epsPlot(df, k, 1, True)
    #dbscan(df, 0.37, 15, True)

    #df_noise = df[df['cluster'] == -1]
    #df_noise = df_noise.drop(columns=['cluster'])
    #print(df_noise['Velocity'].describe())
    #df_noise.to_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/noise_day.csv', index=False, sep=';')


def Threshold(df_tot, df_noise, feature):
    old_mean = np.mean(df_tot[feature])
    old_std = df_tot[feature].std()
    normalized = np.mean(df_noise[feature])
    standard = -1 + df_noise[feature].std()
    # find the interquartile range to measure the variability
    # iqr is a better measure for this than std for skewed data
    q3, q1 = np.percentile(df_noise[feature], [75, 25])
    iqr = -1 + q3 - q1
    print(normalized, iqr)

    df_standardized = Standardize(df_tot, ['Velocity', 'Temp', 'angle', 'Altitude', 'n_lambs', 'age'])
    x_max = max(df_standardized[feature])
    x_min = min(df_standardized[feature])

    antinorm = ((normalized + 1) * (x_max - x_min)/2) + x_min
    anti_iqr = ((iqr + 1) * (x_max - x_min)/2) + x_min
    antistd = ((standard + 1) * (x_max - x_min)/2) + x_min
    threshold = antinorm * old_std + old_mean
    # real value iqr becomes the pluss/minus variability about the mean
    pm = anti_iqr * old_std + old_mean
    stand = antistd * old_std + old_mean

    print(threshold, pm, stand)


#runner(df1)
Threshold(df1, df8, 'Temp')






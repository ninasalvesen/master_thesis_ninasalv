import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

"""
# Elbow method of finding number of clusters for optimal kmeans
sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1)
    df1['clusters'] = kmeans.labels_  # .labels_ er det samme som .predict når man putter inn samme datasett i predict
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
plt.figure(figsize=(16, 8))
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('Elbow method for finding number of clusters in Kmeans++', fontsize=35)
plt.xlabel('Number of clusters', fontsize=25)
plt.ylabel('SSE', fontsize=25)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


print('------')
"""
kmeans = KMeans(n_clusters=6, n_init=5, max_iter=300, tol=1e-04, random_state=None, init='k-means++')
kmeans.fit(df1)
centroids = kmeans.cluster_centers_
print(centroids)
clusters = kmeans.labels_  # samme som kmeans.predict(df1)
df1['cluster'] = clusters
print(df1)

fig1 = px.scatter_3d(df1, x='sin_time', y='cos_time', z='Haversine', size='XY', color='cluster', opacity=1)
#fig1.update_traces(marker_size=2.5)
fig1.update_layout(title='3D cluster rep. Tingvoll')
fig1.show()
"""


# try dimensionality reduction:
pca = PCA(n_components=2)
pca.fit(df1)
df1_reduced = pd.DataFrame(pca.transform(df1), columns=['PC1', 'PC2'])
df1_reduced['cluster'] = clusters
print(df1_reduced)

fig2 = px.scatter_3d(df1_reduced, x='sin_time', y='cos_time', z='Haversine', color='cluster')
fig2.update_traces(marker_size=2.5)
fig2.update_layout(title='2D cluster rep. Tingvoll')
fig2.show()

# eventuelt denne versjonen:
cl0 = df1_reduced.loc[df1_reduced['cluster'] == 0]
cl1 = df1_reduced.loc[df1_reduced['cluster'] == 1]
cl2 = df1_reduced.loc[df1_reduced['cluster'] == 2]
cl3 = df1_reduced.loc[df1_reduced['cluster'] == 3]
print(cl0)
print(cl2)

plt.figure(figsize=(10, 10))
plt.scatter(cl0['PC1'], cl0['PC2'], c='b')
plt.scatter(cl1['PC1'], cl1['PC2'], c='r')
plt.scatter(cl2['PC1'], cl2['PC2'], c='c')
plt.scatter(cl3['PC1'], cl3['PC2'], c='y')
plt.show()




# litt div greier av visualisering jeg har samlet opp
#y_kmeans = kmeans.fit_predict(df[['Haversine', 'sin_time', 'cos_time']])
#kmeans.fit(df[['Haversine', 'sin_time', 'cos_time']])

#fig = px.scatter_matrix(df1, width=1400, height=1400, title='Scatter matrix of features in Tingvoll')
#fig.show()



#plt.scatter(df['cos_time'], df['Haversine'])
#plt.show()

#fig1 = px.scatter(df, x="sin_time", y="cos_time", size="Haversine")
#fig1.update_layout(title="3 Features Representation")
#fig1.show()

#fig2 = px.scatter_3d(df1, x="sin_time", y="cos_time", z="Haversine")
#fig2.update_layout(title="3 Features Representation 3D")
#fig2.show()

#clusters = pd.DataFrame(df, columns=df.columns)
#clusters['label'] = kmeans.labels_
#polar = clusters.groupby("label").mean().reset_index()
#polar = pd.melt(polar, id_vars=["label"])
#fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=1400)
#fig4.show()


# line polar / polar plot seaborn
# finne hvilke verdier i Kmeans man bør kjøre som gir best resultat
# lage fine grafer
# finne en måte å måle performance på clusteringen?
# finne hvilke tider på døgnet sin/cos-parene tilsvarer gitt avgrensingen i grafen (cl0.cos_max/min = ?, dette burde ha overlappende verdier i hver cluster)
"""

import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#sns.set_style('darkgrid')

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv',
                 delimiter=';', low_memory=False)

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')

df1.dropna(subset=['Datetime'], inplace=True)
df1.reset_index(inplace=True, drop=True)
df1 = df1.drop(columns=['Datetime', 'Lat', 'Lon', 'X', 'Y', 'Data set', 'uniq.log', 'Farm', 'minutes'])

print(df1)
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


df1 = Standardize(df1, ['Haversine'])
df1 = Normalize(df1, ['Haversine'])
print(df1)
print(df1.describe())

"""
# Elbow method of finding number of clusters for optimal kmeans
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df1)
    #data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()
"""

print('------')

kmeans = KMeans(n_clusters=4, n_init=5, max_iter=300, tol=1e-04, random_state=None, init='k-means++')
kmeans.fit(df1)
centroids = kmeans.cluster_centers_
print(centroids)
clusters = kmeans.predict(df1)
df1['cluster'] = clusters
print(df1)

"""
pca = PCA(n_components=2)
pca.fit(df1)
df1_reduced = pd.DataFrame(pca.transform(df1), columns=['PC1', 'PC2'])
df1_reduced['cluster'] = clusters
print(df1_reduced)

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
"""

cl0 = df1.loc[df1['cluster'] == 0]
cl1 = df1.loc[df1['cluster'] == 1]
cl2 = df1.loc[df1['cluster'] == 2]
cl3 = df1.loc[df1['cluster'] == 3]
fig2 = px.scatter_3d(df1, x='sin_time', y='cos_time', z='Haversine', color='cluster')
fig2.update_traces(marker_size=2.5)
fig2.update_layout(title="3D cluster rep. Tingvoll")
fig2.show()


"""
plt.figure(figsize=(8, 8))
plt.scatter(df1[:, 0], df1[:, 1], c=model.labels_.astype(float))
plt.show()





fig = plt.figure(figsize=(16, 10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
x = df1['sin_time']
y = df1['cos_time']
z = df1['Haversine']
sc = ax.scatter(x, y, z, s=50, marker='o', alpha=1)
cluster_center1 = ax.scatter(0.00982075, -0.76518385, 0.25713714, marker='o', c='r')
cluster_center2 = ax.scatter(0.01065463, 0.82746879, -0.00152551, marker='o', c='r')
cluster_center3 = ax.scatter(0.01019984, -0.41410931, -0.71671186, marker='o', c='r')
ax.set_xlabel('sin_time', fontsize=20, labelpad=15)
ax.set_ylabel('cos_time', fontsize=20, labelpad=15)
ax.set_zlabel('Haversine', fontsize=20, labelpad=15)
ax.set_title('Feature dependency Tingvoll', fontsize=40, pad=30)
plt.show()
"""

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


#line polar / polar plot seaborn
# how to show results of the clustering, a graph or just numbers?

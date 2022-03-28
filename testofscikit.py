import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V6 sinetime.csv',
                 delimiter=';', low_memory=False)

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

df.dropna(subset=['Datetime'], inplace=True)
df.reset_index(inplace=True, drop=True)
df = df.drop(columns=['Datetime', 'Lat', 'Lon', 'X', 'Y', 'Data set', 'uniq.log', 'Farm', 'minutes'])

print(df)

#plt.scatter(df['cos_time'], df['Haversine'])
#plt.show()

kmeans = KMeans(n_clusters=3, n_init=5, max_iter=300, tol=1e-04, random_state=42)
#y_kmeans = kmeans.fit_predict(df[['Haversine', 'sin_time', 'cos_time']])
kmeans.fit(df[['Haversine', 'sin_time', 'cos_time']])

#fig = px.scatter_matrix(df, width=1200, height=1600)
#fig.show()


#fig1 = px.scatter(df, x="sin_time", y="cos_time", size="Haversine")
#fig1.update_layout(title="3 Features Representation")
#fig1.show()

#fig2 = px.scatter_3d(df, x="sin_time", y="cos_time", z="Haversine")
#fig2.update_layout(title="3 Features Representation")
#fig2.show()

clusters = pd.DataFrame(df, columns=df.columns)
clusters['label'] = kmeans.labels_
polar = clusters.groupby("label").mean().reset_index()
polar = pd.melt(polar, id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=1400)
fig4.show()

#scale haversine
#line polar
# how to show results of the clustering, a graph or just numbers?
# how to plot the data without plotly and by using seaborn the same way or by not getting a html figure?

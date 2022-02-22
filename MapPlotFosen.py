import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V4 med Haversine.csv',
                 delimiter=';')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df.head())

lat = []
long = []
time = []
dataset = 2
check = 0
i = 0
while i < len(df):
    if pd.isnull(df.at[i, 'Datetime']):
        i += 1
        check += 1

    if dataset == check:
        time.append(df.at[i, 'Datetime'])
        lat.append(df.at[i, 'Lat'])
        long.append(df.at[i, 'Lon'])
        while pd.isnull(df.at[i, 'Data set']):
            lat.append(df.at[i+1, 'Lat'])
            long.append(df.at[i+1, 'Lon'])
            i += 1
            if i == (len(df) - 1):
                time.append(df.at[i-1, 'Datetime'])
                break
        time.append(df.at[i-1, 'Datetime'])
        break
    i += 1

print("Time frame for the time spent on pastures: ", time)
extreme_values = [min(long), max(long), min(lat), max(lat)]
print("extreme values: ", extreme_values)


figure1 = plt.imread(f"/Users/ninasalvesen/Documents/Sauedata/Bilder/figure_fosen_close.png")
fig1 = [10.1856, 10.4740, 63.6959, 63.8074]
figure2 = plt.imread(f"/Users/ninasalvesen/Documents/Sauedata/Bilder/figure_fosen_far.png")
fig2 = [10.1720, 10.4604, 63.7823, 63.8934]

fig, ax = plt.subplots()
ax.scatter(long, lat, zorder=1, alpha=0.6, c="b", s=1)
ax.imshow(figure1, zorder=0, extent=fig1, aspect="auto")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid("True")
plt.show()

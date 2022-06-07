import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

"""
sns.set_style('darkgrid')

df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V3 med Haversine.csv',
                  delimiter=';', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%Y-%m-%d %H:%M:%S')
print(df1)

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V5 after cut 2.0.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%Y-%m-%d %H:%M:%S')

print(df2)


df1 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Tingvoll data/Samlet data Tingvoll V5 after cut 4.0.csv', delimiter=';',
                  dtype={"Initial start": "str", "Start": "str", "Stop": "str"})

df1['Datetime'] = pd.to_datetime(df1['Datetime'], format='%d/%m/%Y %H:%M')

df2 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Fosen_Telespor/Samlet data Fosen V6 after cut 4.0.csv',
                  delimiter=';')
df2['Datetime'] = pd.to_datetime(df2['Datetime'], format='%d/%m/%Y %H:%M')

df3 = pd.read_csv('/Users/ninasalvesen/Documents/Sauedata/Datasett_ferdig/Endelig/Total.csv',
                  delimiter=';', dtype={"Initial start": "str", "Start": "str", "Stop": "str"})
df3['Datetime'] = pd.to_datetime(df3['Datetime'], format='%Y-%m-%d %H:%M:%S')
df3.rename(columns={'Velocity':'Haversine'}, inplace=True)
"""


def FindExtremeDates(df):
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Date'] = df_copy['Datetime'].dt.date.astype('str').str[-5:]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    # set an arbitrary year to make it a DateTime object, but it is only m and d that we are interested in
    min_date = '2000-' + df_copy['Date'].min()
    max_date = '2000-' + df_copy['Date'].max()
    return min_date, max_date


def DateActivity(df):
    min_date, max_date = FindExtremeDates(df)
    date_range = pd.date_range(start=min_date, end=max_date).astype('str').str[-5:].tolist()
    activity = np.zeros(len(date_range))
    freq = np.zeros(len(date_range))
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Date'] = df_copy['Datetime'].dt.date.astype('str').str[-5:]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    df_copy.reset_index(inplace=True, drop=True)

    i = 0
    while i < len(df_copy):
        activity[date_range.index(df_copy.at[i, 'Date'])] += df_copy.at[i, 'Haversine']
        freq[date_range.index(df_copy.at[i, 'Date'])] += 1
        i += 1
    for k in range(len(activity)):
        if freq[k] != 0:
            activity[k] = activity[k] / freq[k]
    date_range = pd.to_datetime(date_range, format='%m-%d')
    return date_range, activity


def DateActivityPerYear(df, year):
    j = 0
    start = 0
    temp = 0
    while j < len(df):
        while df.at[j, 'Datetime'].year == year and j != len(df) - 1:
            if temp == 0:
                start = j - 1
            j += 1
            temp += 1
            if pd.isnull(df.at[j, 'Datetime']):
                j += 1
        if temp != 0:
            break
        j += 1
    if year == 2020:
        j += 2
    df_copy = df.loc[start:j-2, :].copy()

    min_date, max_date = FindExtremeDates(df)
    date_range = pd.date_range(start=min_date, end=max_date).astype('str').str[-5:].tolist()
    activity = np.zeros(len(date_range))
    freq = np.zeros(len(date_range))
    df_copy.loc[:, 'Date'] = df_copy.loc[:, 'Datetime'].dt.date.astype('str').str[-5:]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    df_copy.reset_index(inplace=True, drop=True)

    i = 0
    while i < len(df_copy):
        activity[date_range.index(df_copy.at[i, 'Date'])] += df_copy.at[i, 'Haversine']
        freq[date_range.index(df_copy.at[i, 'Date'])] += 1
        i += 1
    for k in range(len(activity)):
        if freq[k] != 0:
            activity[k] = activity[k] / freq[k]
    date_range = pd.to_datetime(date_range, format='%m-%d')
    return date_range, activity


def ActivityPerHour(df):
    hours = []
    for i in range(24):
        hours.append(i)
    activity = np.zeros(len(hours))
    freq = np.zeros(len(hours))
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Hour'] = df_copy['Datetime'].astype('str').str[-8:-6]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    df_copy.reset_index(inplace=True, drop=True)
    df_copy['Hour'] = df_copy['Hour'].astype('int')

    i = 0
    while i < len(df_copy):
        activity[hours.index(df_copy.at[i, 'Hour'])] += df_copy.at[i, 'Haversine']
        freq[hours.index(df_copy.at[i, 'Hour'])] += 1
        i += 1
    for k in range(len(activity)):
        activity[k] = activity[k] / freq[k]
    hours = [str(int) for int in hours]
    for q in range(len(hours)):
        if len(hours[q]) == 1:
            hours[q] = '0' + hours[q] + ':00'
        else:
            hours[q] = hours[q] + ':00'
    return hours, activity


def ActivityPerHourBoxPlot(df):
    hours = [[0 for _ in range(1)] for _ in range(24)]
    df_copy = df.copy()  # make a copy so that the original dataframe is not altered
    df_copy['Hour'] = df_copy['Datetime'].astype('str').str[-8:-6]
    df_copy.dropna(subset=['Datetime'], inplace=True)
    df_copy.reset_index(inplace=True, drop=True)
    df_copy['Hour'] = df_copy['Hour'].astype('int')

    i = 0
    while i < len(df_copy):
        hours[df_copy.at[i, 'Hour']].append(df_copy.at[i, 'Haversine'])
        i += 1
    for i in range(len(hours)):
        del(hours[i][0])
    return hours


"""
dates1, activity1 = DateActivity(df1)
dates2, activity2 = DateActivity(df2)

# Plot of activity per date in Tingvoll
fig1, ax1 = plt.subplots(figsize=(16, 8))
plt.bar(x=dates1, height=activity1)
ax1.set_xlabel(' ', fontsize=1)
ax1.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax1.set_title('Mean activity per date in Tingvoll', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
date_form = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(date_form)


# Plot of activity per date in Fosen
fig2, ax2 = plt.subplots(figsize=(16, 8))
plt.bar(x=dates2, height=activity2)
ax2.set_xlabel(' ', fontsize=1)
ax2.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax2.set_title('Mean activity per date in Fosen', fontsize=40, pad=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
ax2.xaxis.set_major_formatter(date_form)


dates2_2018, activity2_2018 = DateActivityPerYear(df2, 2018)
dates2_2019, activity2_2019 = DateActivityPerYear(df2, 2019)
dates2_2020, activity2_2020 = DateActivityPerYear(df2, 2020)

# Figure of activity plots for all years and total in Fosen
fig3, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(14, 10))
ax3.bar(x=dates2_2018, height=activity2_2018)
ax3.set_title('2018', fontsize=18)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.set(ylim=(0, 300))
date_form = DateFormatter('%d.%m')
ax3.xaxis.set_major_formatter(date_form)

ax4.bar(x=dates2_2019, height=activity2_2019)
ax4.set_title('2019', fontsize=18)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)
ax4.set(ylim=(0, 300))
ax4.xaxis.set_major_formatter(date_form)

ax5.bar(x=dates2_2020, height=activity2_2020)
ax5.set_title('2020', fontsize=18)
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
ax5.set(ylim=(0, 300))
ax5.xaxis.set_major_formatter(date_form)

ax6.xaxis.set_major_formatter(date_form)
ax6.bar(x=dates2, height=activity2)
ax6.set_title('Mean total', fontsize=18)
ax6.tick_params(axis='x', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)
ax6.set(ylim=(0, 300))
ax6.xaxis.set_major_formatter(date_form)

fig3.suptitle('Mean activity per year in Fosen in m/hr per date', fontsize=30)
plt.tight_layout()
plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/mean_activity_per_date_Fosen_aftercut4.0.png", dpi=500)


dates1_2012, activity1_2012 = DateActivityPerYear(df1, 2012)
dates1_2013, activity1_2013 = DateActivityPerYear(df1, 2013)
dates1_2014, activity1_2014 = DateActivityPerYear(df1, 2014)
dates1_2015, activity1_2015 = DateActivityPerYear(df1, 2015)
dates1_2016, activity1_2016 = DateActivityPerYear(df1, 2016)

# Figure of activity plots for all years and total in Tingvoll
fig4, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3, 2, figsize=(14, 10))
ax7.bar(x=dates1_2012, height=activity1_2012)
ax7.set_title('2012', fontsize=18)
ax7.tick_params(axis='x', labelsize=12)
ax7.tick_params(axis='y', labelsize=12)
ax7.set(ylim=(0, 385))
date_form = DateFormatter('%d.%m')
ax7.xaxis.set_major_formatter(date_form)

ax8.bar(x=dates1_2013, height=activity1_2013)
ax8.set_title('2013', fontsize=18)
ax8.tick_params(axis='x', labelsize=12)
ax8.tick_params(axis='y', labelsize=12)
ax8.set(ylim=(0, 385))
ax8.xaxis.set_major_formatter(date_form)

ax9.bar(x=dates1_2014, height=activity1_2014)
ax9.set_title('2014', fontsize=18)
ax9.tick_params(axis='x', labelsize=12)
ax9.tick_params(axis='y', labelsize=12)
ax9.set(ylim=(0, 385))
ax9.xaxis.set_major_formatter(date_form)

ax10.bar(x=dates1_2015, height=activity1_2015)
ax10.set_title('2015', fontsize=18)
ax10.tick_params(axis='x', labelsize=12)
ax10.tick_params(axis='y', labelsize=12)
ax10.set(ylim=(0, 385))
ax10.xaxis.set_major_formatter(date_form)

ax11.bar(x=dates1_2016, height=activity1_2016)
ax11.set_title('2016', fontsize=18)
ax11.tick_params(axis='x', labelsize=12)
ax11.tick_params(axis='y', labelsize=12)
ax11.set(ylim=(0, 385))
ax11.xaxis.set_major_formatter(date_form)

ax12.bar(x=dates1, height=activity1)
ax12.set_title('Mean total', fontsize=18)
ax12.tick_params(axis='x', labelsize=12)
ax12.tick_params(axis='y', labelsize=12)
ax12.set(ylim=(0, 385))
ax12.xaxis.set_major_formatter(date_form)

fig4.suptitle('Mean activity per year in Tingvoll in m/hr per date', fontsize=30)
plt.tight_layout()
plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/mean_activity_per_date_Tingvoll_aftercut4.0.png", dpi=500)


hours1, hourlyActivity1 = ActivityPerHour(df1)
#hours2, hourlyActivity2 = ActivityPerHour(df2)

# Plot of activity per hour in Tingvoll
fig5, ax13 = plt.subplots(figsize=(16, 8))
plt.bar(x=hours1, height=hourlyActivity1)
ax13.set_xlabel('Hour', fontsize=35, labelpad=20)
ax13.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax13.set_title('Mean activity per hour in Tingvoll', fontsize=40, pad=30)
ax13.set(ylim=(0, 210))
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
ax13.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/mean_activity_per_hour_Tingvoll_aftercut4.0.png", dpi=500)


# Plot of activity per hour in Fosen
fig6, ax14 = plt.subplots(figsize=(16, 8))
plt.bar(x=hours2, height=hourlyActivity2)
ax14.set_xlabel('Hour', fontsize=35, labelpad=20)
ax14.set_ylabel('Velocity, m/hr', fontsize=35, labelpad=30)
ax14.set_title('Mean activity per hour in Fosen', fontsize=40, pad=30)
ax14.set(ylim=(0, 210))
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
ax14.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/after cut 4.0/mean_activity_per_hour_Fosen_aftercut4.0.png", dpi=500)


boxHours1 = ActivityPerHourBoxPlot(df1)
#boxHours2 = ActivityPerHourBoxPlot(df2)
labels = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
          '18', '19', '20', '21', '22', '23']

# Boxplot of activity per hour in Tingvoll
fig7, ax15 = plt.subplots(figsize=(16, 8))
ax15.set_xlabel('Hour', fontsize=30, labelpad=20)
ax15.set_ylabel('Velocity, m/h', fontsize=30, labelpad=20)
ax15.set_title('Mean velocity per hour outliers', fontsize=40, pad=30)
medianProps = dict(linewidth=2.5)
plt.boxplot(boxHours1, showfliers=True, labels=labels, showmeans=True, medianprops=medianProps)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax15.set(ylim=(-10, 17000))
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/boxplot_fliers_velocity_total.png", dpi=500)


# Boxplot of activity per hour Fosen
fig8, ax16 = plt.subplots(figsize=(16, 8))
ax16.set_xlabel('Hour', fontsize=30, labelpad=20)
ax16.set_ylabel('Velocity, m/hr', fontsize=30, labelpad=20)
ax16.set_title('Mean activity per hour in Fosen', fontsize=40, pad=30)
plt.boxplot(boxHours2, showfliers=False, labels=labels, showmeans=True, medianprops=medianProps)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax16.set(ylim=(-15, 500))
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/b4 cut/boxplot_fliers_activity_per_hour_Fosen_b4cut.png", dpi=500)


# violin plot of the activity in bins
boxHours1 = ActivityPerHourBoxPlot(df1)
labels = ['04-09', '10-15', '16-21', '22-03']
bins = [[], [], [], []]
for i in range(len(boxHours1)):
    if i <= 3 or i >= 22:
        bins[3].extend(boxHours1[i])
    if i <= 9 and i >= 4:
        bins[0].extend(boxHours1[i])
    if i <= 15 and i >= 10:
        bins[1].extend(boxHours1[i])
    if i <= 21 and i >= 16:
        bins[2].extend(boxHours1[i])

fig8, ax16 = plt.subplots(figsize=(16, 8))
ax16.set_xlabel('Hour', fontsize=30, labelpad=20)
ax16.set_ylabel('Velocity, m/h', fontsize=30, labelpad=20)
ax16.set_title('Mean velocity per hour with distribution', fontsize=40, pad=30)
plt.violinplot(bins, showextrema=False, showmeans=True)
plt.xticks(fontsize=20, ticks=[1, 2, 3, 4], labels=labels)
plt.yticks(fontsize=20)
ax16.set(ylim=(-10, 1000))
plt.tight_layout()
#plt.savefig("/Users/ninasalvesen/Documents/Sauedata/Bilder/Master/01 Total/activity_violinplot.png", dpi=500)
"""

plt.show()



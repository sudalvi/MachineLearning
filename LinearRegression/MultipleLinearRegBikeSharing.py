import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection

data = pd.read_csv('bike_sharing.csv')
print(data.shape)
data.head()
data.dtypes
data.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

data = data.drop(['instant','dteday','yr'], axis=1)
data['season'] = data.season.astype('category')
data['month'] = data.month.astype('category')
data['hour'] = data.hour.astype('category')
data['holiday'] = data.holiday.astype('category')
data['weekday'] = data.weekday.astype('category')
data['workingday'] = data.workingday.astype('category')
data['weather'] = data.weather.astype('category')

data.dtypes
data.isnull().any()
fix, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'count',
                       'weekday']],
            x='hour', y='count',
            hue='weekday', ax=ax)
ax.set(title="Use of the system during weekdays and weekends")

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'casual',
                       'weekday']],
            x='hour', y='casual',
            hue='weekday', ax=ax)
ax.set(title="Use of the system by casual users")

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'registered',
                       'weekday']],
            x='hour', y='registered',
            hue='weekday', ax=ax)
ax.set(title="Use of the system by registered users")

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'count',
                       'weather']],
            x='hour', y='count',
            hue='weather', ax=ax)
ax.set(title="Use of the system: weather condition")

fig, ax = plt.subplots(figsize=(20,10))
sn.pointplot(data=data[['hour',
                       'count',
                       'season']],
            x='hour', y='count',
            hue='season', ax=ax)
ax.set(title="Use of the system: season")

fig, ax = plt.subplots(figsize=(20,10))
sn.barplot(data=data[['weekday',
                      'count']],
            x='weekday', y='count')
ax.set(title="Daily distribution")

plt.show()
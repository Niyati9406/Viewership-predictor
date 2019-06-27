# viewership-predictor

#Problem Statement: Analysing and predicting “Why the viewership of digital streaming shows is lessening”. 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

media = pd.read_csv('mediacompany.csv')
media = media.drop('Unnamed: 7',axis = 1)


media.head()


media['Date'] = pd.to_datetime(media['Date'])

media.head()

from datetime import date

d0 = date(2017, 2, 28)
d1 = media.Date
delta = d1 - pd.to_datetime(d0)
media['day']= delta
media.head()

media['day'] = media['day'].astype(str)
media['day'] = media['day'].map(lambda x: x[0:2])
media['day'] = media['day'].astype(int)

media.plot.line(x='day', y='Views_show')

colors = (0,0,0)
area = np.pi*3
plt.scatter(media.day, media.Views_show, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel("Day")
host.set_ylabel("View_Show")
par1.set_ylabel("Ad_impression")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(media.day,media.Views_show, color=color1,label="View_Show")
p2, = par1.plot(media.day,media.Ad_impression,color=color2, label="Ad_impression")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

par2.spines['right'].set_position(('outward', 60))      

par2.xaxis.set_ticks([])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')

media['weekday'] = (media['day']+3)%7
media.weekday.replace(0,7, inplace=True)
media['weekday'] = media['weekday'].astype(int)
media.head()

X = media[['Visitors','weekday']]

y = media['Views_show']

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X,y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)

import statsmodels.api as sm 
X = sm.add_constant(X)

lm_1 = sm.OLS(y,X).fit()
print(lm_1.summary())
        
def cond(i):
    if i % 7 == 5: return 1
    elif i % 7 == 4: return 1
    else :return 0
    return i

media['weekend']=[cond(i) for i in media['day']]
media.head()

X = media[['Visitors','weekend']]
y = media['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)

lm_2 = sm.OLS(y,X).fit()
print(lm_2.summary())
        
X = media[['Visitors','weekend','Character_A']]
y = media['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)

lm_3 = sm.OLS(y,X).fit()
print(lm_3.summary())
        
media['Lag_Views'] = np.roll(media['Views_show'], 1)
media.Lag_Views.replace(108961,0, inplace=True)
media.head(10)

X = media[['Visitors','Character_A','Lag_Views','weekend']]
y = media['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)
lm_4 = sm.OLS(y,X).fit()
print(lm_4.summary())

plt.figure(figsize = (20,10))       
sns.heatmap(media.corr(),annot = True)

X = media[['weekend','Character_A','Views_platform']]

y = media['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)

lm_5 = sm.OLS(y,X).fit()
print(lm_5.summary())


X = media[['weekend','Character_A','Visitors']]

y = media['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_6 = sm.OLS(y,X).fit()
print(lm_6.summary())
        
X = media[['weekend','Character_A','Visitors','Ad_impression']]

y = media['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)

lm_7 = sm.OLS(y,X).fit()
print(lm_7.summary())
        
X = media[['weekend','Character_A','Ad_impression']]

y = media['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_8 = sm.OLS(y,X).fit()
print(lm_8.summary())

media['ad_impression_million'] = media['Ad_impression']/1000000


X = media[['weekend','Character_A','ad_impression_million','Cricket_match_india']]

y = media['Views_show']

import statsmodels.api as sm 
X = sm.add_constant(X)

lm_9 = sm.OLS(y,X).fit()
print(lm_9.summary())


X = media[['weekend','Character_A','ad_impression_million']]


y = media['Views_show']


import statsmodels.api as sm 
X = sm.add_constant(X)

lm_10 = sm.OLS(y,X).fit()
print(lm_10.summary())


X = media[['weekend','Character_A','ad_impression_million']]
X = sm.add_constant(X)
Predicted_views = lm_10.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(media.Views_show, Predicted_views)
r_squared = r2_score(media.Views_show, Predicted_views)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)            
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Views', fontsize=16)                               


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Views_show-Predicted_views', fontsize=16)                


X = media[['weekend','Character_A','Visitors']]
X = sm.add_constant(X)
Predicted_views = lm_6.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(media.Views_show, Predicted_views)
r_squared = r2_score(media.Views_show, Predicted_views)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Views', fontsize=16)                               


c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Views_show-Predicted_views', fontsize=16) 

















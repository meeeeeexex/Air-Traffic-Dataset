import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_linear
import sklearn.cluster as sk_cluster
import numpy as np
import sklearn.preprocessing as sk_preproc

def set_print_opt():
    pd.set_option('display.max_columns',None)
    pd.set_option('display.width',None)
    pd.options.display.max_rows = 999
set_print_opt()

df = pd.read_csv('data/Air_Traffic_Passenger_Statistics.csv')

our_df = pd.DataFrame(data=df['Activity Period'])
our_df=our_df.applymap(lambda x:'01.0' +str(x%100)+ '.'+ str(x//100))
our_df['From'] = df['GEO Region']
our_df['Passengers']=df['Passenger Count']
our_df['Action'] = df['Activity Type Code']

filter_deplane = df['Activity Type Code']=='Deplaned'
filter_enplane = df['Activity Type Code']=='Deplaned'
filter_europe = df['GEO Region'] == 'Europe'

plt.figure(figsize=(15, 8),dpi=80)

deplaned_df = our_df.loc[filter_deplane]
enplaned_df = our_df.loc[filter_enplane]

print(our_df.columns)

sum_deplaned = sum(deplaned_df['Passengers'])
sum_enplaned = sum(enplaned_df['Passengers'])


amount_of_usa = len(our_df['From']=='US')

X = our_df[['Passengers']]
X_from =our_df[['From']]
X_set = set(X_from['From'])

grouped_df = X_from.groupby(by=X_from['From'])
categorical_counrties = {'Mexico':0,'Middle East':1,'Central America':2,'US':3,'Europe':4}
nn = dict()
n = [(lambda x: x) (x) for x in X_set]
cnt = 0
for i in n:
    nn[i] = cnt
    cnt+=1
print(nn)
our_df['categorical country'] = pd.Series((lambda x: nn[x])(x) for x in our_df['From'])

# # plot_colors = ['black','blue','brown','coral','green','gray','yellow','magenta','Olive']
# # pshe = our_df[our_df[['From']]=='US']
# # pshe.dropna(subset=['From'],inplace=True)
#
# print(X_set)
list_of_lens = []
for i in range(9):
    list_of_lens.append(len(our_df[our_df['categorical country']==i]))
print(list_of_lens)

#plt.bar(n,list_of_lens)



europe_df = df.loc[filter_europe]
europe_df = europe_df.drop(columns=['GEO Summary','GEO Region','Price Category Code',
                                    'Price Category Code','Terminal','Boarding Area','Adjusted Activity Type Code',
                                    'Adjusted Passenger Count','Year',
                                    'Month','Activity Type Code', 'Activity Period'])
countries = ['Belarus','France','Turkey','United','Virgin']


airlines = set(europe_df['Operating Airline'])
#попробовтаь решить как задачу классификации, обучив на паре примеров
print(airlines)
airlines = sorted(list(airlines))
#USA,Swiss,Britain,Servisair -swiss, SAS - Sweden,Pacific Aviation' - usa, France,Germany,Belarus, United airlines = USA,Iceland
#Danmark,ireland,turkey,Germany
dataframe = pd.DataFrame()
dataframe['airline'] = pd.Series ((lambda x: airlines[x])(x)for x in range (len(airlines)))
l = ['Ireland','Germany','France','Belarus','Britain','Iceland','Danmark','Germany',
     'US','Sweden','Swiss','Swiss','Swiss','Turkey','Britain','Britain','US','US','France']

dataframe['country'] = l

# model_res = country_model.predict(X)
# dataframe['prediction'] = model_res
print(dataframe)
set_of_countries = list(set(dataframe['country']))

d = {}
for i in range(len(set_of_countries)):
    d[set_of_countries[i]] = i

print(d)
dataframe_categorical = pd.DataFrame(data=pd.Series(np.arange(19)),columns=['numeric airline'])
series_with_num_data = pd.Series(data=((lambda x: d[x])(x)for x in dataframe['country']))


print(series_with_num_data)
dataframe_categorical['numeric country'] = series_with_num_data
print(dataframe_categorical)

X = dataframe_categorical[['numeric airline']]
y = dataframe_categorical[['numeric country']]
X_scaler = sk_preproc.StandardScaler(X)
y_scaler  =sk_preproc.StandardScaler(y)

country_model = sk_linear.LinearRegression()
country_model.fit(X,y)

dataframe_categorical['predicted'] = country_model.predict(X)
#plt.show()

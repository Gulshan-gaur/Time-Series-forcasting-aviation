import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
df=pd.read_csv('AviationData.csv', engine='python')
df['Year'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").year)
df['Month'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").month)
df['Day'] = df['Event.Date'].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").day)
df = df.drop(['Event.Id','Investigation.Type','Accident.Number','Location','Country','Airport.Code','Airport.Name','Injury.Severity','Aircraft.Damage','Registration.Number','FAR.Description','Schedule','Purpose.of.Flight','Air.Carrier','Report.Status','Publication.Date','Make','Model'], axis='columns')
df1=df
df1[['Aircraft.Category']]=df[['Aircraft.Category']].fillna('N1')

df1[['Amateur.Built']]=df[['Amateur.Built']].fillna('N2')
df1[['Engine.Type']]=df[['Engine.Type']].fillna('N3')
df1[['Weather.Condition']]=df[['Weather.Condition']].fillna('N4')
df1[['Broad.Phase.of.Flight']]=df[['Broad.Phase.of.Flight']].fillna('N5')
 


#null_columns=df1.columns[df1.isnull().any()]
#print(df1[null_columns].isnull().sum())

one_hot=pd.get_dummies(df1['Aircraft.Category'])
df1=df1.drop('Aircraft.Category',axis=1)
one_hot=one_hot.drop('Unknown',axis=1)
df1=df1.join(one_hot)

one_hot1=pd.get_dummies(df1['Amateur.Built'])
df1=df1.drop('Amateur.Built',axis=1)
df1=df1.join(one_hot1)

one_hot2=pd.get_dummies(df1['Engine.Type'])
df1=df1.drop('Engine.Type',axis=1)
one_hot2=one_hot2.drop('Unknown',axis=1)
df1=df1.join(one_hot2)

one_hot3=pd.get_dummies(df1['Weather.Condition'])
df1=df1.drop('Weather.Condition',axis=1)
df1=df1.join(one_hot3)

one_hot4=pd.get_dummies(df1['Broad.Phase.of.Flight'])
df1=df1.drop('Broad.Phase.of.Flight',axis=1)
one_hot4=one_hot4.drop('UNKNOWN',axis=1)
df1=df1.join(one_hot4)


df1["Latitude"].fillna(df1["Latitude"].mean(), inplace=True)
df1["Longitude"].fillna(df1["Longitude"].mean(), inplace=True)
df1["Number.of.Engines"].fillna(df1["Number.of.Engines"].median(), inplace=True)
df1["Total.Fatal.Injuries"].fillna(0, inplace=True)
df1["Total.Serious.Injuries"].fillna(0, inplace=True)
df1["Total.Minor.Injuries"].fillna(0, inplace=True)
df1["Total.Uninjured"].fillna(0, inplace=True)

df1.rename(columns={'Total.Fatal.Injuries': 'tfi', 'Total.Serious.Injuries': 'tsi','Total.Minor.Injuries':'tmi'}, inplace=True)

TotalInjuries = [ row.tfi + row.tsi + row.tmi for index, row in df1.iterrows() ]
df1['TotalInjuries'] = TotalInjuries

df1=df1.drop('tfi',axis=1)
df1=df1.drop('tmi',axis=1)
df1=df1.drop('tsi',axis=1)
df1=df1.drop('Total.Uninjured',axis=1)

df1.to_csv('aviationdf1.csv')


df1=pd.read_csv('aviationdf1.csv', engine='python')
df2=df1.filter(['Event.Date','TotalInjuries'],axis=1)
df2.to_csv('aviationdf1arima.csv')




#CODE FOR ARIMA
from pandas import Series
series = Series.from_csv('aviationdf1arima.csv', header=0)

trainsetar1, testsetar1 = series[76013:80013], series[80013:]
print('trainsetar1 %d, testsetar1 %d' % (len(trainsetar1), len(testsetar1)))
trainsetar1.to_csv('trainsetar1.csv')
testsetar1.to_csv('testsetar1.csv')

from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = Series.from_csv('trainsetar1.csv')
series.head()

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]



# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.d, Expected=%.d' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)

from pandas import Series
from matplotlib import pyplot

pyplot.figure()
pyplot.plot(train)

pyplot.xlabel('Days')
pyplot.ylabel('Accidents')
pyplot.show()

pyplot.figure()
pyplot.scatter(np.arange(0,train.shape[0]),train[:])
pyplot.ylabel('Accidents')
pyplot.xlabel('Days')
pyplot.show()

series = Series.from_csv('trainsetar1.csv')
series.plot()
pyplot.show()

'''
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
 # invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 # evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse
 # evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s RMSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
 # load dataset
series = Series.from_csv('dataset.csv')
# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values) '''




from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
# load data
series = Series.from_csv('trainsetar1.csv')
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()



'''
from statsmodels.tsa.arima_model import ARIMA
import numpy
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
series = Series.from_csv('trainsetar1.csv')
X = series.values
X = X.astype('float32')
months_in_year = 12
diff = difference(X, months_in_year)
model = ARIMA(diff, order=(0,0,1))
model_fit = model.fit(trend='nc', disp=0)
bias = 0
# save model
model_fit.save('model1.pkl')
numpy.save('model_bias1.npy', [bias])'''



from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
# load and prepare datasets
dataset = Series.from_csv('trainsetar1.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12
validation = Series.from_csv('testsetar1.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model1.pkl')
bias = numpy.load('model_bias1.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.d, Expected=%.d' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(0,0,1))
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.d, Expected=%.d' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.figure(figsize=(12,16))
pyplot.plot(predictions, color='red',linewidth=2)
pyplot.show()


mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y,linewidth=0.5)
pyplot.plot(predictions, color='red',linewidth=0.5)
pyplot.show()





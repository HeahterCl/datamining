import pandas as pd 
import numpy as np
import time
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg,dtrain,predictors,target,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
	    xgb_param = alg.get_xgb_params()
	    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
	    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics = 'rmse',early_stopping_rounds=early_stopping_rounds)
	    alg.set_params(n_estimators=cvresult.shape[0])
	    print(cvresult)

	print("Starting training")
	#Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain[target], eval_metric = 'rmse')
	print("Finished training")

	#Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	
	#Print model report:
	print ("\nModel Report")
	print ("Accuracy : %.4g" % metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
	
	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
	plt.show()

def replace(replace_list):
	replace_category = replace_list.astype('category')
	categories = replace_category.cat.categories
	replace_categories = np.arange(categories.shape[0])
	replace_category.cat.categories = replace_categories.tolist()
	return replace_category.astype('float64')

def processTime(timestamp):
	timestamp_array = []
	for i in range(timestamp.shape[0]):
		timestamp_array.append(time.mktime(time.strptime(train.timestamp.iloc[i],'%Y/%m/%d')))
	return timestamp_array

if __name__ == '__main__':
	trainFilePath = './../train_macro.csv'
	testFilePath = './../test_macro.csv'
	#macroFilePath = './../macro.csv'

	train = pd.read_csv(trainFilePath)
	test = pd.read_csv(testFilePath)
	#macro = pd.read_csv(macroFilePath)
	target = 'price_doc'
	IDCol = 'id'
	train_features = [x for x in train.columns if x not in [target, IDCol]]
	test_features = [x for x in test.columns if x not in [IDCol]]
	#macro_features = [x for x in macro.columns if x not in [IDCol]]

	#preprocess train data
	#train_macro = pd.merge(train, macro[macro_features], on='timestamp', how='left')
	#train_macro_features = [x for x in train_macro.columns if x not in [target,IDCol]]
	#train_macro.dropna()
	object_columns = []
	for i in range(train.shape[1]):
		if (train.iloc[:,i].dtypes == 'object' and train.iloc[:,i].name != 'timestamp'):
			object_columns.append(i)

	for i in object_columns:
		train.iloc[:,i] = replace(train.iloc[:,i])

	object_columns = []
	for i in range(test.shape[1]):
		if (test.iloc[:,i].dtypes == 'object'):
			object_columns.append(i)

	for i in object_columns:
		test.iloc[:,i] = replace(test.iloc[:,i])

	#train.timestamp = processTime(train.timestamp)
	print("Finished peocess, and trainging soon will begin")

	param = {
		'min_child_weight':np.arange(1,6,2)
	}
	xgbParams = XGBRegressor(
		learning_rate =0.1,
		n_estimators=282,
		max_depth=5,
		min_child_weight=3,
		gamma=0,
		reg_alpha = 100,
		subsample=0.8,
		colsample_bytree=0.7,
		objective= 'reg:linear',
		base_score = 0.5,
		nthread=-1,
		scale_pos_weight=1,
		silent = 0
	)

	#xgbParams = {'booster':'gbtree','objective':'reg:linear','gamma':0,'max_depth':5,'lambda':100,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':3,'eta':0.1}
	dtrain = xgb.DMatrix(train[train_features].values, label=np.log1p(train[target].values))
	watchlist = [(dtrain,'train')]
	model = xgb.train(xgbParams.get_xgb_params(), dtrain, 5000, watchlist, early_stopping_rounds = 50)
	model.save_model('./../sberbank.model')
	#model = xgb.Booster({'nthread':-1})
	#model.load_model('./../sberbank.model')
	dtest = xgb.DMatrix(test[test_features].values)
	pred = model.predict(dtest)
	answer = pd.DataFrame({'id':test[IDCol],'price_doc':np.expm1(pred)})
	answer.to_csv('./../answer.csv')
	#gsearch = GridSearchCV(estimator = xgbParams, param_grid = param,scoring = 'neg_mean_squared_error', cv=5)
	#gsearch.fit(train[train_features], train[target])
	#print("best_params: %s"%gsearch.best_params_)
	#print('best_score: %s'%gsearch.best_score_)
	#modelfit(xgbParams, train, train_features, target)
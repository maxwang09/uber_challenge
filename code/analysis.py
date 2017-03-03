# -*- encoding:utf-8 -*-

import re
import csv
import json
import time
import fileinput
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_category_feature(data, fieldname, titles=('Start Drive','Not Started'), limit=30):
	keys = map(lambda x:x[0],sorted([(key,data[True].get(key,0)+data[True].get(key,0)) for key in \
									set(data[True].keys())|set(data[False].keys()) if key!='NA'],\
						  			key=lambda x:x[1],reverse=True)[:limit])
	N = len(keys); width = 0.35
	D1, D2 = [[data[label].get(key,0) for key in keys]for label in (True, False)]
	fig, ax = plt.subplots()
	rects1 = ax.barh(np.arange(N)+width/2, D1, width, color='r')
	rects2 = ax.barh(np.arange(N)+width/2, D2, width, left=D1, color='b')
	ax.set_title(fieldname)
	ax.set_xlabel('Count'); ax.set_xlim(xmax=max(D1)+max(D2)*1.1)
	ax.set_yticks(np.arange(N)+width); ax.set_yticklabels(tuple(keys))
	ax.legend((rects1[0], rects2[0]), titles)
	plt.savefig('../figure/{0}.png'.format(fieldname))
	for TITLE, BOOL in zip(titles,(True,False)):
		print '{0}\t{1}\t{2}'.format(fieldname,TITLE,json.dumps(data[BOOL]))


def plot_timeline_feature(data, fieldname, titles=('Start Drive','Not Started')):
	keys = sorted(filter(lambda x:0<x<10**4,list(set(data[True].keys())|set(data[False].keys()))))
	D1, D2 = [[data[label].get(key,0) for key in keys]for label in (True, False)]
	plt.figure()
	line1, = plt.plot(keys, D1, 'r-', linewidth=2)
	line2, = plt.plot(keys, D2, 'b-', linewidth=2)
	plt.title(fieldname)
	plt.ylabel('Count'); plt.ylim(ymax=max(D1+D2)*1.1)
	plt.legend((line1, line2), titles)
	plt.savefig('../figure/{0}.png'.format(fieldname))
	for TITLE, BOOL in zip(titles,(True,False)):
		print '{0}\t{1}\t{2}'.format(fieldname,TITLE,json.dumps(data[BOOL]))


def analysis(filename, mode='plot'):
	'''
		Analysis
		TBD: change any date to 'NA' if this date is later than first_trip_date.
	'''
	X, y = [], []; feature_set = {}; data_set = {}

	def get_category_feature(row, fieldname):
		field = row[fieldname] or 'NA'; feature_set[fieldname] = feature_set.get(fieldname,[])
		if not field in feature_set[fieldname]: feature_set[fieldname].append(field)
		return feature_set[fieldname].index(field)

	def get_time_year_feature(row, fieldname, thisyear=2016, defaultvalue=(0,-10**4)):
		get_feature = lambda t: (t, thisyear-t)
		return get_feature(int(row[fieldname])) if row[fieldname].isdigit() else defaultvalue

	get_time = lambda s: None if not re.match(u'/'.join([ur'[0-9]{,2}']*3),s) else time.strptime(s,'%m/%d/%y')

	def get_time_date_feature(row, fieldname, defaultvalue=(-10**4,-10**4)):
		get_feature = lambda t: (t.tm_yday, t.tm_wday)
		return get_feature(get_time(row[fieldname])) if get_time(row[fieldname]) else defaultvalue

	def get_time_date_delta_feature(row, fieldname1, fieldname2, defaultvalue=10**4):
		time1, time2 = map(lambda fieldname:get_time(row[fieldname]), (fieldname1,fieldname2))
		return time1.tm_yday-time2.tm_yday if time1 and time2 else defaultvalue

	with open(filename,'rU') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			city_name, signup_os, signup_channel, vehicle_model, vehicle_make = \
				[get_category_feature(row,fieldname) for fieldname in ('city_name','signup_os','signup_channel','vehicle_model','vehicle_make')]
			((vehicle_year, vehicle_year_delta),) = \
				[get_time_year_feature(row,fieldname) for fieldname in ('vehicle_year',)]
			((signup_date_yd, signup_date_wd), (bgc_date_yd, bgc_date_wd), (vehicle_added_date_yd, vehicle_added_date_wd),) = \
				[get_time_date_feature(row,fieldname) for fieldname in ('signup_date','bgc_date','vehicle_added_date')]
			signup_date_to_bgc_date, bgc_date_to_vehicle_added_date, signup_date_to_vehicle_added_date = \
				[get_time_date_delta_feature(row, t1, t2) for t1, t2 in [('bgc_date','signup_date'),('vehicle_added_date','bgc_date'),('vehicle_added_date','signup_date')]]
			label = bool(get_time(row['first_completed_date']))

			X.append([city_name, signup_os, signup_channel, vehicle_model, vehicle_make, vehicle_year, vehicle_year_delta, \
					  signup_date_yd, signup_date_wd, bgc_date_yd, bgc_date_wd, vehicle_added_date_yd, vehicle_added_date_wd, \
					  signup_date_to_bgc_date, bgc_date_to_vehicle_added_date, signup_date_to_vehicle_added_date])
			y.append(label)

			for fieldname, field in [('city_name',city_name),('signup_os',signup_os),('signup_channel',signup_channel),('vehicle_model',vehicle_model),('vehicle_make',vehicle_make)]:
				data_set[fieldname] = data_set.get(fieldname,{True:{},False:{}})
				data_set[fieldname][label][feature_set[fieldname][field]] = data_set[fieldname][label].get(feature_set[fieldname][field],0)+1
			for fieldname, field in [('vehicle_year',vehicle_year),('signup_date_yd',signup_date_yd),('signup_date_wd',signup_date_wd),\
																   ('bgc_date_yd',bgc_date_yd),('bgc_date_wd',bgc_date_wd),\
																   ('vehicle_added_date_yd',vehicle_added_date_yd),('vehicle_added_date_wd',vehicle_added_date_wd),\
																   ('signup_date_to_bgc_date',signup_date_to_bgc_date),\
																   ('bgc_date_to_vehicle_added_date',bgc_date_to_vehicle_added_date),\
																   ('signup_date_to_vehicle_added_date',signup_date_to_vehicle_added_date)]:
				data_set[fieldname] = data_set.get(fieldname,{True:{},False:{}})
				data_set[fieldname][label][field] = data_set[fieldname][label].get(field,0)+1
	
	if mode == 'plot':
		for fieldname, fielddata in data_set.iteritems():
			if fieldname in ('city_name','signup_os','signup_channel','vehicle_model','vehicle_make'):
				plot_category_feature(fielddata,fieldname)
			if fieldname in ('vehicle_year','signup_date_yd','signup_date_wd','bgc_date_yd','bgc_date_wd','vehicle_added_date_yd','vehicle_added_date_wd',\
							 'signup_date_to_bgc_date','bgc_date_to_vehicle_added_date','signup_date_to_vehicle_added_date'):
				plot_timeline_feature(fielddata,fieldname)
	elif mode == 'predict':
		return np.array(X), np.array(y)
	else:
		raise Exception('Mode not supported.')


def prediction(filename, classifier='DecisionTree', get_feature_importance=False):
	'''
		Prediction
	'''
	from sklearn import svm
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.linear_model import SGDClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import KFold
	from sklearn.metrics import f1_score, confusion_matrix

	if classifier == 'SVM_rbf':
		clf = svm.SVC(kernel='rbf')
	elif classifier == 'SVM_linear':
		clf = svm.SVC(kernel='linear')
	elif classifier == 'DecisionTree':
		clf = DecisionTreeClassifier()
	elif classifier == 'AdaBoost':
		clf = AdaBoostClassifier()
	elif classifier == 'RandomForest':
		clf = RandomForestClassifier()
	elif classifier == 'GradientBoosting':
		clf = GradientBoostingClassifier()
	elif classifier == 'SGDClassifier_L1':
		clf = SGDClassifier(loss="hinge", penalty="l1")
	elif classifier == 'SGDClassifier_L2':
		clf = SGDClassifier(loss="hinge", penalty="l2")
	elif classifier == 'LogisticRegression_L1':
		clf = LogisticRegression(penalty="l1")
	elif classifier == 'LogisticRegression_L2':
		clf = LogisticRegression(penalty="l2")
	else:
		raise Exception('Classifer not supported.')

	X, y = analysis(filename, mode='predict')
	if get_feature_importance:
		clf.fit(X,y)
		print list(clf.feature_importances_)
	else:
		kf = KFold(n_splits=4); yp = []
		for train, test in kf.split(X):
			clf.fit(X[train],y[train])
			yp.extend(clf.predict(X[test]))
		print classifier, f1_score(y,yp)
		print classifier, confusion_matrix(y,yp)


if __name__ == '__main__':
	analysis('../data/ds_challenge_v2_1_data.csv')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='SVM_rbf')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='SVM_linear')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='DecisionTree')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='AdaBoost')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='RandomForest')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='GradientBoosting')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='SGDClassifier_L1')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='SGDClassifier_L2')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='LogisticRegression_L1')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='LogisticRegression_L2')
	prediction('../data/ds_challenge_v2_1_data.csv', classifier='GradientBoosting', get_feature_importance=True)
	pass

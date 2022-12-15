# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

def Create_Models():
	# loading dataset to pandas dataframe
	sonar_data = pd.read_csv('/datasets/sonar_data.csv',header=None)
	sonar_data[60] = sonar_data[60].replace({'M':0,'R':1})

	"""
	M [1] ---> Mine
	R [0] ---> Rock 
	"""
	X = sonar_data.drop(columns=60,axis=1)
	Y = sonar_data[60]
	
	x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=1)

	"""Model Training """

	lg_model = LogisticRegression()
	dt_model = DecisionTreeClassifier()
	rf_model = RandomForestClassifier()
	sv_model = SVC()
	nb_model = GaussianNB()
	kn_model = KNeighborsClassifier()

	print(lg_model.fit(x_train,y_train))
	print(dt_model.fit(x_train,y_train))
	print(rf_model.fit(x_train,y_train))
	print(sv_model.fit(x_train,y_train))
	print(nb_model.fit(x_train,y_train))
	print(kn_model.fit(x_train,y_train))

	print(lg_model.score(x_test,y_test))
	print(dt_model.score(x_test,y_test))
	print(rf_model.score(x_test,y_test))
	print(sv_model.score(x_test,y_test))
	print(nb_model.score(x_test,y_test))
	print(kn_model.score(x_test,y_test))

	print(cross_val_score(lg_model,X,Y,cv=3))
	print(cross_val_score(dt_model,X,Y,cv=3))
	print(cross_val_score(rf_model,X,Y,cv=3))
	print(cross_val_score(sv_model,X,Y,cv=3))
	print(cross_val_score(nb_model,X,Y,cv=3))
	print(cross_val_score(kn_model,X,Y,cv=3))

	# Model Evaluation
	# accuracy on traing data
	X_train_prediction = lg_model.predict(x_train)
	training_data_acc = accuracy_score(X_train_prediction,y_train)

	print("Acc on train data : {}".format(training_data_acc))

	# Model Evaluation
	# accuracy on test data
	X_test_prediction = lg_model.predict(x_test)
	test_data_acc = accuracy_score(X_test_prediction,y_test)

	print("Acc on test data : {}".format(test_data_acc))
	"""Model Creation"""
	joblib.dump(lg_model,'/models/lg_model.sav')
	joblib.dump(dt_model,'/models/dt_model.sav')
	joblib.dump(rf_model,'/models/rf_model.sav')
	joblib.dump(sv_model,'/models/sv_model.sav')
	joblib.dump(nb_model,'/models/nb_model.sav')
	joblib.dump(kn_model,'/models/kn_model.sav')


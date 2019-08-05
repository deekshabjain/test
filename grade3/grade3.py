import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('grades.csv')

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 9].values

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, :-1])
x[:,:-1 ] = imputer.transform(x[:, :-1])


from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import ExtraTreesClassifier
model1 = ExtraTreesClassifier()
model1.fit(x,y)
feat_importances = pd.Series(model1.feature_importances_)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

from sklearn.externals import joblib
joblib.dump(regressor,'model/model.pkl')

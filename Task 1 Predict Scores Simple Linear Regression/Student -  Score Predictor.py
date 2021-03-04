import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

data = pd.read_csv(url)
print(data.head())
print(data.info())
print(data.describe())

# fairly small dataset
# no missing values detected

"""Separating independent and dependent variables"""
X = data.iloc[:,0].values
X = X.reshape(-1,1)
y = data.iloc[:,1].values


"""Exploratory Data Analysis"""
#pandas plotting
from pandas.plotting import scatter_matrix
columns = ["Hours","Scores"]
scatter_matrix(data[columns],figsize=(12,6))
plt.figure(clear="True")
#matplotlib plotting
plt.scatter(x = X,y = y)
plt.xlabel("Hours of Study")
plt.xlabel("Student Scores")

"""from these plots we can see that we have a linear relationship between
X('Hours') and y('Scores')"""
# checking for correlation between X and y
correlation_matrix = data.corr()
y_correlation = correlation_matrix["Scores"].sort_values(ascending = False)
"""here we can see that we have a strong positive correlation of 0.97
 between X and Y"""
 
# Splitting our data in train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = 0.2,
                                                 random_state = 0)
"""Data Preprocessing"""
# no missing values
# no need of feature scaling
# from EDA we can see that we can use linear regression
"""Modeling """
#trying different models(score results in different models scores.txt)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
"""we will do hyper parameter tuning later if needed"""
"""Training the model"""
model = LinearRegression()
# model = SVR(kernel = "rbf")
# model = DecisionTreeRegressor(criterion = "mse",random_state = 42)
#model = RandomForestRegressor(n_estimators=10,random_state = 42)
model.fit(x_train,y_train)

"""Plotting the trained model"""
line = model.coef_ *x_train + model.intercept_
plt.figure(clear = True)
plt.scatter(x_train,y_train)
plt.plot(x_train,line)
plt.title("Trained Model")
plt.show()

"""Evaluating the model"""
from sklearn.model_selection import cross_val_score
eval_scores = cross_val_score(model,
                              x_train,y_train,
                              scoring = "neg_mean_squared_error",
                              cv = 10)
eval_scores = np.sqrt(-eval_scores)
scores_mean = eval_scores.mean()
scores_std = eval_scores.std()

"""Saving the best model"""
# Out of all models we evaluated linear regression performs best
from joblib import dump,load
dump(model,'score_predictor.joblib')



"""Testing the model"""
model = load('score_predictor.joblib')
from sklearn.metrics import mean_squared_error

y_pred = model.predict(x_test)

#comparing actual test values vs predicted values
test_df = pd.DataFrame(data ={"Actual":y_test,"Predicted":y_pred} 
                       )
rmse_test = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"RMSE on test set is: {rmse_test}")
"""Predicting the score if student studies 9.25 hr/day"""
student_score = model.predict([[9.25]])
print(student_score.shape)
print(f'The student score will be {student_score[0]} if he/she studies 9.25 hrs ')

"""Conclusion"""
# 1. if we had more data our model could have performed better then this
# 2. for a very small dataset our predictions are great


 




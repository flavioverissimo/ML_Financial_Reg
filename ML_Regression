# packages
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


TRAIN_MODEL = True 
PRINT_DATA = False

try:
	# reading model file
	with open("model/pred_investiment_regression.pkl", 'rb') as md:
		model_loaded = pickle.load(md)

	# reading r2_score file
	with open("model/r2_score.txt", 'rb') as score:
		r2_score_loaded = pickle.load(score)

	# checking if the r2_score is good
	if r2_score_loaded >= 0.6:
		TRAIN_MODEL = False
		getValue = float(input("How much money do you want to invest here? (write the value): "))
		getValue_array = np.array([getValue])
		getValue_array = getValue_array.reshape(-1,1)
		predicted_value = model_loaded.predict(getValue_array)
		print("\nTotal Investiment: ", getValue)
		print("The predicted return is: ", predicted_value[0])
except:
	print("\nThere are not any files to read due to the r2_score being less than 0.6 or because the program is running for the first time.\n\n")
	

try:
	if TRAIN_MODEL:
		# loading data
		data = pd.read_csv("data/dataset.csv")

		# separating data in x and y
		x = data.iloc[:, 0:-1].values
		y = data.iloc[:, 1].values

		# separating data for train and test
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

		# defining model
		model = LinearRegression()

		# training model
		model.fit(x_train, y_train)

		# doing prediction with test data
		y_predicted = model.predict(x_test)

		# Model Avaliation
		# Mean Absolute Error
		MAE = mean_absolute_error(y_test, y_predicted)

		# Mean Squared Error and Root Mean Squared Error
		MSE = mean_squared_error(y_test, y_predicted)
		RMSE = math.sqrt(MSE)

		# R2 Score
		r2_score = r2_score(y_test, y_predicted)

		# showing the values
		print("Mean Absolute Error: ", MAE)
		print("Mean Squared Error and Root Mean Squared Error", MSE, RMSE)
		print("R2 Score", r2_score)


		# saving model as a file
		if r2_score >= 0.6:
			pkl_path = "model/pred_investiment_regression.pkl"
			r2score = "model/r2_score.txt"

			with open(pkl_path, 'wb') as path:
				pickle.dump(model, path)

			with open(r2score, 'wb') as score:
				pickle.dump(r2_score, score)


		# showing the data with plt 
		if PRINT_DATA:

			# defining the y = a * x + b
			regressionLine = model.coef_ * x + model.intercept_

			# showing data
			plt.scatter(x, y)
			plt.title("Investment x Return")
			plt.xlabel("Investment")
			plt.ylabel("Return")
			plt.plot(x, regressionLine, color="orange")
			plt.show()

			
			# comparison y_test
			y_comparison = pd.DataFrame({ "Real Values": y_test, "Predicted Values": y_predicted})

			fix, ax = plt.subplots()
			index = np.arange(len(x_test))
			bar_with = 0.40
			current = plt.bar(index, y_comparison["Real Values"], bar_with, label="Real Values")
			predicted = plt.bar(index, y_comparison["Predicted Values"], bar_with, label="Predicted Values") 
			plt.title("Real Value x Predicted Value")
			plt.xlabel("Investiment")
			plt.ylabel("Predicted Return")
			plt.legend()
			plt.show()
except:
	print("There was some error during the execution!")


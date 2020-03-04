import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder,PolynomialFeatures, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import FunctionTransformer
from sklearn_pandas import CategoricalImputer
from sklearn.dummy import DummyRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
import copy
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import pickle
import xgboost as xgb

cars = pd.read_csv("train-data.csv")

cars = cars.dropna(subset = ["Power", "Mileage", "Engine", "Seats"])

del cars['New_Price']

cars["Seats"] = cars["Seats"].astype(int)

cars = cars[cars.Power != 'null bhp']

cars["Price"] = cars["Price"]*100000/54

cars["Brand"] = cars["Name"].str.split().str[0]

cars["Brand"] = cars["Brand"].str.lower()

cars["Age"] = 2020 - cars["Year"]

def clean(column):
    cars[column] = cars[column].str.split(" ").str[0]

clean("Mileage")
clean("Engine")
clean("Power")

cars["Mileage"] = cars["Mileage"].str.split(".").str[0].astype(int)
cars["Power"] = cars["Power"].str.split(".").str[0].astype(int)
cars["Engine"] = cars["Engine"].astype(int)

y = cars['Price']

X = cars.drop(['Name', "Price", 'Unnamed: 0', 'Year'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

mapper = DataFrameMapper([
    ('Location', LabelBinarizer()),
    (['Kilometers_Driven'], StandardScaler()),
    ('Fuel_Type', LabelBinarizer()),
    ('Transmission', LabelEncoder()),
    ('Owner_Type', LabelBinarizer()),
    (['Mileage'], StandardScaler()),
    (['Engine'], StandardScaler()),
    (['Power'], StandardScaler()),
    ('Seats', None),
    ('Brand', LabelBinarizer()),
    ('Age', None),
], df_out=True)

cars = mapper.fit_transform(cars)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)

data_dmatrix = xgb.DMatrix(data=Z_train,label=y_train)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.5,
                max_depth = 25, alpha = 25, n_estimators = 50)

xg_reg.fit(Z_train,y_train)

xg_reg.score(Z_train,y_train)

xg_reg.score(Z_test,y_test)

pipe = make_pipeline(mapper, xg_reg)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open("pipe.pkl", "wb"))

del pipe

pipe = pickle.load(open("pipe.pkl", "rb"))

new_data = pd.DataFrame({
    'Location': ['Mumbai'],
    'Kilometers_Driven': [72000],
    'Fuel_Type': ['CNG'],
    'Transmission': ['Manual'],
    'Owner_Type': ['First'],
    "Mileage": [26],
    'Engine': [998],
    'Power': [58],
    'Seats': [4],
    'Brand': ['Maruti'],
    'Age': [5]

})

prediction = pipe.predict(new_data)
np.round(prediction, 2)

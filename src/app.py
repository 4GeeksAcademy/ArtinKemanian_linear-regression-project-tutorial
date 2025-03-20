from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

datos = datos.drop_duplicates().reset_index(drop = True)

datos["sex_n"] = pd.factorize(datos["sex"])[0]
datos["smoker_n"] = pd.factorize(datos["smoker"])[0]
datos["region_n"] = pd.factorize(datos["region"])[0]
num_variables = ["age", "bmi", "children", "sex_n", "smoker_n", "region_n", "charges"]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(datos[num_variables])
datos_scal = pd.DataFrame(scal_features, index = datos.index, columns = num_variables)

X = datos_scal.drop("charges", axis = 1)
y = datos_scal["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selection_model = SelectKBest(f_regression, k = 4)
selection_model.fit(X_train, y_train)

selected_columns = X_train.columns[selection_model.get_support()]
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = selected_columns)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = selected_columns)

X_train_sel["charges"] = y_train.values
X_test_sel["charges"] = y_test.values
X_train_sel.to_csv("data/processed/train_limpio.csv", index = False)
X_test_sel.to_csv("data/processed/test_limpio.csv", index = False)

datos_train = pd.read_csv("data/processed/train_limpio.csv")
datos_test = pd.read_csv("data/processed/test_limpio.csv")

fig, axis = plt.subplots(4, 2, figsize = (10, 14))
datos = pd.concat([datos_train, datos_test])

sns.regplot(data = datos, x = "age", y = "charges", ax = axis[0, 0])
sns.heatmap(datos[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(data = datos, x = "bmi", y = "charges", ax = axis[0, 1])
sns.heatmap(datos[["charges", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1], cbar = False)

sns.regplot(data = datos, x = "children", y = "charges", ax = axis[2, 0])
sns.heatmap(datos[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0], cbar = False)

sns.regplot(data = datos, x = "smoker_n", y = "charges", ax = axis[2, 1])
sns.heatmap(datos[["charges", "smoker_n"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1], cbar = False)

plt.tight_layout()
plt.show()

X_train = datos_train.drop(["charges"], axis = 1)
y_train = datos_train["charges"]
X_test = datos_test.drop(["charges"], axis = 1)
y_test = datos_test["charges"]

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"Intercep (a): {modelo.intercept_}")
print(f"Coefficients (b1, b2): {modelo.coef_}")

y_pred = modelo.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
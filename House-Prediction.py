import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)
data=data.dropna()
skewed_feats = data.skew().sort_values(ascending=False) 
high_skew = skewed_feats[abs(skewed_feats) > 0.75]
for col in high_skew.index:
    data[col] = np.log1p(data[col]) 
X = data.drop("medv", axis=1)  
y = data["medv"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)    
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nFirst 10 predictions:\n", comparison.head(20))
print("R² Score (Accuracy):",r2*100,"%")
plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='cyan')
plt.show()
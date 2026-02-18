import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("homeprices.csv")

new_df = df.drop('price', axis='columns')

reg = LinearRegression()

reg.fit(new_df.values, df.price.values)

# Making Predictions
print(reg.predict([[5000]]))
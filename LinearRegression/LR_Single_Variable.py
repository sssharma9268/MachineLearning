import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("./LinearRegression/areas.csv")
#print(df.head())
plt.figure()
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price,color='red',marker='+')
#plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
print(reg.predict([[3300]]))
print(reg.coef_) #it prints the value of m in equation y=mx+b
print(reg.intercept_)#it prints the value of b

d = pd.read_csv("./LinearRegression/areas_predict.csv")
p = reg.predict(d)
print(p)
d['prices'] = p
#print(d)
#d.to_csv("./LinearRegression/areas_predicted.csv",index=False)

plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()
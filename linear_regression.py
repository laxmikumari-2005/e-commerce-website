
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Data(Hours studied)
X=np.array([[1],[2],[3],[4],[5],[6]])
#Data(Scores obtained)
y=np.array([10,20,30,40,50,60])

#Creating a linear regression model
model=LinearRegression()
model.fit(X,y)

#Predicting the score for 6 hours of study
predicted_score=model.predict([[6]])

#plot
plt.scatter(X,y,color='blue',label='Data points')
plt.plot(X,model.predict(X),color='red',label='Regression line')

#highlighting predicted point
plt.scatter(6,predicted_score,color='green',label='Predicted point')


plt.title('simple linear regression')
plt.xlabel('Hours studied')
plt.ylabel('Scores obtained')
plt.legend()
plt.show()

print("Predicted score for 6 hours of study:", predicted_score[0])
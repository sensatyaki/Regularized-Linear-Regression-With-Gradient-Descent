
L2 gradient Descent vs L2 closed Form:
---------------------------------------

L2 gradient descent gave results close to L2 closed from solution when the value of alpha (learning factor of the Data) is very less  and threshold error which have been used for the convergence the gradient descent function is 1e6 or 1e7.But gradient descent takes much time to converge. When the error and the learning factor was reduced the gradient was able to converge faster giving in a increase in RMSE based on alpha and error threshold.

Feature Enginering which I have done:
---------------------------------------
Feature Engineering gives better results compared taking the features directly. Adding features, root of the features given (power 0.5) and  squares (power 2) gave less RMSE when the predicted output.

Normalization on the data is done to make the gradient converge faster. 


Remark:
---------------------------------------
60% of the given data has been taken for training, and the rest of 40% data taken for cross validation
If we limit the data for training overfitting gets reduced.
If the weights are calculated on the entire given data,RMSE becomes very less for testing data which I have taken from training data.
but the RMSE of the predicted ouput based on the test data on kaggle showed a significant difference, which is due to the overfitting.

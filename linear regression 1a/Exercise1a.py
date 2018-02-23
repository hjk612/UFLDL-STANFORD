import numpy as np
import scipy.optimize
from sklearn.metrics import mean_squared_error

#defining the path
file_path = '/Users/Hatim/Downloads/housing.data.txt'

#loading the data
data = np.loadtxt(file_path)

#inserting a column of one for the bias/intercept term
data = np.insert(data,0,1,axis=1)

#randomly shuffling the data
np.random.shuffle(data)

#train and test data
X_train = data[:400,:-1]
y_train = data[:400,-1]

X_test = data[400:,:-1]
y_test = data[400:,-1]

def objective_function(theta,X,y):
    #writing in matrix form J = (y - X.theta)'(y - X.theta)
    func = (y.T - X.dot(theta)).T.dot(y.T-X.dot(theta))
    return func

def gradient(theta,X,y):
    func = 2*X.T.dot(X.dot(theta))- 2*X.T.dot(y.T)
    return func

J_route=[]
optimization = scipy.optimize.minimize(fun=objective_function,
                                       x0 = np.random.rand(14),
                                       args=(X_train,y_train),
                                       jac=gradient,
                                       method ='bfgs',
                                       options={'maxiter':200,'disp':True},
                                       callback=lambda x:J_route.append(objective_function(x,X_train,y_train)))


optimal_theta = optimization.x

y_train_predicted = X_train.dot(optimal_theta)
y_test_predicted = X_test.dot(optimal_theta)

train_rmse = np.sqrt(mean_squared_error(y_train,y_train_predicted))
test_rmse = np.sqrt(mean_squared_error(y_test,y_test_predicted))


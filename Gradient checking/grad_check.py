import numpy as np

#defining the path
file_path = 'housing.data.txt'

#loading the data
data = np.loadtxt(file_path)

#inserting a column of one for the bias/intercept term
data = np.insert(data,0,1,axis=1)

#randomly shuffling the data
np.random.shuffle(data)

#train and test data
X_train = data[:400,:-1]
y_train = data[:400,-1]


def objective_function_linear(theta,X,y):
    #writing in matrix form J = (y - X.theta)'(y - X.theta)
    func = (y.T - X.dot(theta)).T.dot(y.T-X.dot(theta))
    return func

def gradient_linear(theta,X,y):
    func = 2*X.T.dot(X.dot(theta))- 2*X.T.dot(y.T)
    return func

epsilon = 1e-4
theta = np.random.rand(X_train.shape[1])
gradient_true = gradient_linear(theta,X_train,y_train)
error =[]
for i in range(len(theta)):
    #making copies of theta for theta+ and theta-
    #make sure to make deep copy else changing theta+ or theta- will change original theta
    theta_plus = np.copy(theta)
    theta_minus = np.copy(theta)
    #incrementing the i-th dimension by epsilon
    theta_plus[i] = theta[i] + epsilon
    #decreasing the i-th dimension by epsilon
    theta_minus[i] = theta[i] - epsilon
    #getting the derivative wrt to theta-i (along one dimesion of theta)
    gradient_approx = (objective_function_linear(theta_plus,X_train,y_train) -objective_function_linear(
            theta_minus,X_train,y_train))/(2*epsilon)
    #appending the absolute difference between derivative by approximation and by true function 
    #across one dimension of theta
    error.append(np.abs(gradient_approx-gradient_true[i]))
    
print("average error = "+str(np.average(error)))

#same steps can be implemented for logistic regression

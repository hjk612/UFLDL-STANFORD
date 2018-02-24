import numpy as np
import scipy.optimize
from sklearn.metrics import accuracy_score

#importing idx2numpy package to read the MNIST data
import idx2numpy 
raw_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')
raw_train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
raw_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
raw_test_labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

#converting the 3d array to 2d array
raw_train = raw_train.reshape(raw_train.shape[0],raw_train.shape[1]**2)
raw_test = raw_test.reshape(raw_test.shape[0],raw_test.shape[1]**2)

#selecting only the '1' and '0' images/data
train = raw_train[raw_train_labels<2].astype('float64')
test = raw_test[raw_test_labels<2].astype('float64')

#normalize the features for faster convergence
def normalize_features(data):
    mean = np.mean(data,axis = 0)
    std = np.std(data,axis = 0)
    
    #adding 0.1 to avoid division by 0
    norm_data = (data - mean)/(std+0.1)
    return norm_data

train = normalize_features(train)
test = normalize_features(test)

#concatenating the features(pixels) and label
train = np.concatenate((train,
                        raw_train_labels[raw_train_labels<2].reshape(12665,1)),axis = 1)
test = np.concatenate((test,
                        raw_test_labels[raw_test_labels<2].reshape(2115,1)),axis = 1)

#adding 1s for the intercept
train = np.insert(train,0,1,axis=1)
test = np.insert(test,0,1,axis=1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def objective_function(theta,X,y):
    logistic = sigmoid(X.dot(theta))
    func = y.dot(np.log(logistic)) +(1-y).dot(np.log(1-logistic))
    return -func


def gradient(theta,X,y):
    term1 = sigmoid(X.dot(theta))-y
    return term1.dot(X)

J_history = []
res = scipy.optimize.minimize(
    fun=objective_function,
    x0=np.random.rand(train.shape[1]-1) * 0.001,
    args=(train[:,:-1], train[:,-1]),
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: J_history.append(objective_function(x, train[:,:-1], train[:,-1])),
)

train_predicted_prob = sigmoid(train[:,:-1].dot(res.x))
test_predicted_prob = sigmoid(test[:,:-1].dot(res.x))

def prob2pred(x):
    x[x>0.5] = 1
    x[x<0.5] = 0
    return x

train_pred = prob2pred(train_predicted_prob)
test_pred = prob2pred(test_predicted_prob)

train_acc = accuracy_score(train[:,-1],train_pred)
test_acc = accuracy_score(test[:,-1],test_pred)
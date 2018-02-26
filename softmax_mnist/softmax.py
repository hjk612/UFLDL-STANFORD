import numpy as np
import os
import scipy.optimize
import idx2numpy
from sklearn.metrics import accuracy_score

raw_train = idx2numpy.convert_from_file('train-images-idx3-ubyte')
raw_train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
raw_test = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
raw_test_labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

#converting the 3d array to 2d array
train = raw_train.reshape(raw_train.shape[0],raw_train.shape[1]**2)
test = raw_test.reshape(raw_test.shape[0],raw_test.shape[1]**2)


#normalize the features for faster convergence
def normalize_features(data):
    mean = np.mean(data,axis = 0)
    std = np.std(data,axis = 0)
    
    #adding 0.1 to avoid division by 0
    norm_data = (data - mean)/(std+0.1)
    return norm_data

X_train = normalize_features(train)
X_test = normalize_features(test)

#adding 1s for the intercept
X_train = np.insert(X_train,0,1,axis=1)
X_test = np.insert(X_test,0,1,axis=1)

#one hot encoding the labels. Better for matrix multiplication
n_label = 10
y_train = np.eye(n_label)[raw_train_labels]
y_test = np.eye(n_label)[raw_test_labels]

#model preparation
m, n = X_train.shape
theta = np.random.rand(n,n_label)

def probability(theta,X,y):
    if len(theta.shape)==1:
        theta = theta.reshape(n,n_label)
    temp1 = np.exp(X.dot(theta))
    normalize_term = np.sum(temp1,axis=1)
    prob = temp1/normalize_term.reshape(X.shape[0],1)
    return prob

def objective_function(theta,X,y):
    log_prob = np.log(probability(theta,X,y))
    final = log_prob*y
    return -np.sum(final)

def gradient(theta,X,y):
    prob = probability(theta,X,y)
    final = y - prob
    return -X.T.dot(final).flatten()

J_history = []

res = scipy.optimize.minimize(
    fun=objective_function,
    x0=np.random.rand(785,10)*0.01,
    args=(X_train,y_train),
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: J_history.append(objective_function(x, X_train, y_train)),
)
optimal_theta = res.x.reshape(n,n_label)

#train data predictions
train_pred = np.argmax(X_train.dot(optimal_theta),axis=1)

#test data predictions
test_pred = np.argmax(X_test.dot(optimal_theta),axis=1)

acc_train = accuracy_score(train_pred,raw_train_labels)
acc_test = accuracy_score(test_pred,raw_test_labels)

print("Train accuracy: "+str(acc_train))
print("\nTest accuracy: "+str(acc_test))
    
    

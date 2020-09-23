import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn import svm, model_selection

train_df = pd.read_csv("studentspen-train.csv")
train_df
test = pd.read_csv("studentsdigits-test.csv").to_numpy()

X = train_df.iloc[:,0:8]
y = train_df.iloc[:,8]
print(X)
print(y)
X = X.to_numpy()
y = y.to_numpy()

#data for classes 1 and 9
data1 = train_df[train_df['Digit'] == 1] 
data9 = train_df[train_df['Digit'] == 9]
data19 = np.concatenate((data1, data9))

y9 = data19[:,8]
X9 = data19[:,0:8]
y9 = np.reshape(y9, (747,1))

y9_class = np.zeros((len(y9),1))
for i, y in enumerate(y9):
    if y == 9:
        y9_class[i] = -1
    else:
        y9_class[i] = 1.
        
#source problem 
D = 8
C = 2
w_source = cp.Variable((D,1))
P = np.diag(np.ones(D))
eps_s = cp.Variable((len(X9),1))
b_s = cp.Variable()
objective = cp.Minimize(cp.sum_squares(w_source) * 0.5 + cp.sum(cp.square(eps_s)*C))
constraints = [cp.multiply(y9_class, (X9 @ w_source + b_s)) >= 1 - eps_s, eps_s >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var w = {}, b = {}".format(w_source.value, b_s.value))

data7 = train_df[train_df['Digit'] == 7]
data17 = np.concatenate((data1, data7))
data17
y7 = data17[:,8]
X7 = data17[:,0:8]
y7 = np.reshape(y7, (764,1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X7, y7, test_size = 0.2)

y7_class = np.zeros((len(X_train),1))
for i, y in enumerate(y_train):
    if y == 7:
        y7_class[i] = -1
    else:
        y7_class[i] = 1.

#target - no transfer
D = 8
C = 2
w_no = cp.Variable((D,1))
P = np.diag(np.ones(D))
epsilon = cp.Variable((len(X_train),1))
b_no = cp.Variable()
objective = cp.Minimize(cp.sum_squares(w_no) * 0.5 + cp.sum(cp.square(epsilon)*C))
constraints = [cp.multiply(y7_class, (X_train @ w_no + b_no)) >= 1 - epsilon, epsilon >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var w = {}, b = {}".format(w_no.value, b_no.value))

## target - hypothesis transfer
D = 8
C = 2
P = np.diag(np.ones(D))
w_ht = cp.Variable((D, 1))
epsilon = cp.Variable((len(X_train),1))
b_ht = cp.Variable()
objective = cp.Minimize(cp.sum_squares(w_ht) * 0.5 + cp.sum(cp.square(epsilon)*C))
constraints = [cp.multiply(y7_class, (X_train@(w_ht + w_source.value) + b_ht)) >= 1 - epsilon]#, cp.sum_squares((w_source - w)) <= 100]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var w = {}, b = {}".format(w_ht.value, b_ht.value))

#target - instance transfer
#support vectors 
sup = y9_class* (X9.dot(w_source.value) + b_s.value) - 1
eps = 1e-2
sup_v1 = ((-eps<sup) & (sup<eps)).flatten()
sup_vecs = X9[sup_v1]
y_s = y9_class[sup_v1]

D = 8
C = 2
w_i = cp.Variable((D,1))
P = np.diag(np.ones(D))
epsilon = cp.Variable((len(X_train),1))
eps2 = cp.Variable((len(sup_vecs),1))
b_i = cp.Variable()
objective = cp.Minimize(cp.sum_squares(w_i) * 0.5 + cp.sum(cp.square(epsilon)*C) + cp.sum(cp.square(eps2)*C))
constraints = [cp.multiply(y7_class, (X_train @ w_i + b_i)) >= 1 - epsilon, cp.multiply(y_s, (sup_vecs @ w_i + b_i)) >= 1 - eps2 , epsilon >= 0, eps2 >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var w = {}, b = {}".format(w_i.value, b_i.value))

def predict(X, w, b):
    predicted = {}
    for idx, x in enumerate(X):
        res = x.dot(w.value) + b.value
        #print(res)
        if res >= 0:
            predicted[idx] = 1
        else:
            predicted[idx] = 7
    return predicted

def error(pred, y):
    correct = 0
    for i in range(len(pred)):
        if(pred[i] == y[i]):
            correct += 1
    err = 1 - correct / len(pred)
    return err

#error estimates
#no transfer
pred_no = predict(X_test, w_no, b_no)
err_no = error(pred_no, y_test)

#hypothesis transfer
pred_ht = predict(X_test, w_ht + w_source, b_ht)
err_ht = error(pred_ht, y_test)

#instance transfer
pred_i = predict(X_test, w_i, b_i)
err_i = error(pred_i, y_test)

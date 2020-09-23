import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn import svm, model_selection, metrics
import sys
from math import e

train_df = pd.read_csv("studentspen-train.csv")
train_df
test = pd.read_csv("studentsdigits-test.csv").to_numpy()

X = train_df.iloc[:, 0:8]
y = train_df.iloc[:, 8]
print(X)
print(y)
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)

# mapping functions


def phi(y, x):
    p = np.zeros((len(x), 80))
    for idx, xi in enumerate(x):
        start = y[idx]*8
        for i in range(start, start+8):
            p[idx, i] = xi[i - start]
    return p


def phi_single(y, x):
    p = np.zeros((len(x), 80))
    for idx, xi in enumerate(x):
        start = y*8
        for i in range(start, start+8):
            p[idx, i] = xi[i - start]
    return p


def phi_pred(y, x):
    p = np.zeros(80)
    start = y*8
    for i in range(start, start+8):
        p[i] = x[i - start]
    return p


def train(X, y):
    # soft margin constraints
    D = 80
    C = 2
    w = cp.Variable((D, 1))
    b = cp.Variable()
    epsilon = cp.Variable((len(X), 1))
    soft_constraints = []
    count = 0

    for yp in range(10):
        soft_constraints += [(phi(y, X) @ w) >=
                             ((phi_single(yp, X) @ w) + 1 - epsilon)]
    soft_constraints += [epsilon >= 0]
    soft_constraints

    # soft
    objective = cp.Minimize(cp.sum(cp.square(w))*0.5 +
                            cp.sum(cp.square(epsilon)*4))
    prob = cp.Problem(objective, soft_constraints)
    prob.solve(verbose=True)
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var w = {}, b = {}".format(w.value, b.value))

    return w, b


def predict(test, w):
    predicted = np.zeros(len(test), dtype=np.int)
    for idx, x in enumerate(test):
        # print(idx)
        c = 0
        exp = -sys.maxsize - 1  # wt phi(y,x)
        for i in range(10):
            w_pred = np.zeros((80, 1))
            for j in range(i*8, i*8+8):
                w_pred[j] = w.value[j]
            curr = (w_pred.T).dot(phi_pred(i, x))
            # print(w_pre)
            # print(curr)
            if curr > exp:
                exp = curr
                c = i
            # print(c)
        predicted[idx] = c
    return predicted

#cross validation
w_t,b_t = train(X_train, y_train)
predicted_t = predict(X_test, w_t)

correct = 0
for i in range(len(predicted_t)):
    if(predicted_t[i] == y_test[i]):
        correct += 1
error = 1 - correct / len(predicted_t)

#predict test set
w,b = train(X, y)
predicted = predict(X, w)
np.savetxt("haitongli_preds_multiclass.txt", predicted, fmt='%s')

#PAC bound
n = len(X_train)
vc = 10
#pac = (error + 4 * np.log(4/0.05) + 4 * vc * np.log((2*e*n)/vc)) / n
pac1 = error + np.sqrt((vc * (np.log(2*n/vc)+1) + np.log(4/0.05))/n)

#rbf kerneled version - error estimate
clf = svm.SVC(kernel='rbf') 
clf.fit(X_train, y_train)

y_pred_t = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_t))

#rbf kerneled version - predict test set
clf = svm.SVC(kernel='rbf') 
clf.fit(X, y)

y_pred = clf.predict(test)
np.savetxt("haitongli_preds_multiclass_rbfkernel.txt", y_pred, fmt='%s')
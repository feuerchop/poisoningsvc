'''
Created on Mar 28, 2015
This package contains common methods used in poisoning one-class SVM
@author: Xiao, Huang
'''

from joblib import Memory, Parallel, delayed
import numpy as np
from numpy.linalg import norm, lstsq
from scipy.linalg import solve, eigvals
import sklearn.preprocessing as prep
from sklearn.metrics.pairwise import pairwise_kernels as Kernels
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix


def objective(xc, clf, dataset, target='dec_val'):
    """
    Objective function we need to maximise to poisoning purpose
    The key idea here is to have a unified method to describe the
    adversarial objective function.
    :rtype float
    :param xc: the malicious points with size mxd, which we inject into the training set
    :param clf: classifier we try to poison, here is the OCSVM
    :param dataset: a dict contains keys ['train', 'train_labels', 'test', 'test_labels']
    :param type: objective value that we measure (to maximise)
    :return: measured error, e.g., accuracy, MSE, objective value
    """
    Xtr = dataset['train']  # train data is a must
    if dataset.has_key("train_labels"): ytr = dataset['train_labels']
    if dataset.has_key("test"): Xtt = dataset['test']
    if dataset.has_key("test_labels"): ytt = dataset['test_labels']

    # reshape the malicious points into ndarray
    n, d = Xtr.shape
    m = xc.size/d
    xc = xc.reshape(m, d)

    # append the malicious data to form a contaminated dataset
    Xtr_ = np.r_['0,2', Xtr, xc]

    # TODO: update SVC instead of retrain with xc
    clf.fit(Xtr_)  # <---- this is just a lazy update

    if target is 'fval':
        # objective values on untainted dataset
        if type(clf).__name__ is "OCSVM":
            K = Kernels(Xtr_, metric=clf.kernel, filter_params=True, gamma=clf.gamma, coef0=clf.coef0,
                        degree=clf.degree)
            # new set of bounded SVs without the xc
            bsv_ind_new = np.setdiff1d(clf.bsv_ind, [n+1])
            alphas = clf.dual_coef_
            slack_variables = clf.intercept_-clf.decision_function(Xtr_[bsv_ind_new, :])
            # 1/2 |w|^2+1/vl sum\xi - rho
            fval = 0.5*alphas.dot(K[np.ix_(clf.support_, clf.support_)]).dot(
                alphas.T)+slack_variables.sum()-clf.intercept_

        # TODO: we may support other type of objective function
        return fval[0]

    if target is 'clferr':
        # classification error on test set
        if Xtt is not None and ytt is not None:
            y_clf = clf.predict_y(Xtt)
            return 1-acc(ytt, y_clf)
        else:
            print 'You need give the test dataset!'
            return None

    if target is 'dec_val':
        # decision value: w'*x - rho
        return clf.decision_function(Xtr).sum()

    if target is 'fnr':
        # false negative rate (outliers are classified as normal)
        if Xtt is not None and ytt is not None:
            y_clf = clf.predict_y(Xtt)
            cf_max = confusion_matrix(ytt, y_clf, labels=[1, -1])
            return 1-float(cf_max[0, 0])/cf_max[0].sum()
        else:
            print 'You need give the test dataset!'
            return None

    if target is 'fpr':
        # false positive rate (normal are classified as outliers)
        if Xtt is not None and ytt is not None:
            y_clf = clf.predict_y(Xtt)
            cf_max = confusion_matrix(ytt, y_clf, labels=[1, -1])
            return 1-float(cf_max[1, 1])/cf_max[1].sum()
        else:
            print 'You need give the test dataset!'
            return None


def grad_w(xc, clf, dataset):
    """
    gradient of objective W wrt. x_c
    compute the gradient of xc in X
    classifier must be trained on X first! We avoid retraining gradient while
    computing the gradient!
    :param xc:
    :param clf: malicious point, only single-point gradient is supported
    :param X: training set X
    :return: gradient of xc
    """
    # Initialize
    X = dataset['train']
    n, d = X.shape
    xc = xc.reshape(1, d)
    X_ = np.r_[X, xc]

    # fit OCSVM on X_
    clf.fit(X_)

    # TODO: check correctness
    # vector of gradient alpha
    K_x = Kernels(X_, X_[clf.sv_ind], metric=clf.kernel, filter_params=True, gamma=clf.gamma, coef0=clf.coef0, degree=clf.degree)
    K_sv = K_x[clf.sv_ind, :]
    lhs = np.repeat(K_sv[0].reshape(1, K_sv[0].size), clf.sv_ind.size-1, axis=0)-K_sv[1:]
    lhs = np.vstack((lhs, np.ones((1, clf.sv_ind.size))))
    # numerical correction
    lhs = lhs+1e-6*np.eye(lhs.shape[0])
    rhs = np.zeros((clf.sv_ind.size, d))

    # solve the linear system by lstsq
    vs, residuals, rank, s = lstsq(lhs, rhs)
    # vs = solve(lhs, rhs)
    # correct the solution according to KKT(1)
    #     vs[0] = -vs[1:].sum(axis=0)
    #         print 'residuals: ', residuals
    #         print 'rank: %d lhs_rows: %d ' % (rank, clf.sv_ind.size-1)
    random_sv = np.random.choice(clf.sv_ind, 1)
    return (K_x[0:n, :] - np.repeat(K_x[random_sv].reshape(1, clf.sv_ind.size), n, axis=0)).dot(vs).sum(axis=0)


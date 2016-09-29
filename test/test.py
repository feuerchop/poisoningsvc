"""
Created on Mar 27, 2015

@author: Xiao, Huang
"""

import utils
import scipy.io as sio
import scipy.optimize as opt
from LibPoisonOCSVM import *
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit as sss
import sklearn.preprocessing as prep
from OCSVM import *

# X = sio.loadmat('../data/toy_sample.mat')['X']
# y = -sio.loadmat('../data/toy_sample.mat')['y'].ravel()

X, y = utils.gen_noise_gauss(500, cov=.25)

# X, y = gen_svc_data([0, 0], 2.5, cluster_sizes=[100, 20, 10, 10])
# X = prep.scale(X)

cvfolds = sss(y, n_iter=1, test_size=0.85)
data = dict()
for tr_id, tt_id in cvfolds:
    data['train'] = X[tr_id]
    data['test'] = X[tt_id]
    data['train_labels'] = y[tr_id]
    data['test_labels'] = y[tt_id]
n, d = X.shape
clf = OCSVM(nu=0.2, gamma=.5)
clf.fit(data['train'])

big_x, big_y = utils.getBoxbyX(data['train'], grid=30)
big_xy = np.c_[big_x.reshape(big_x.size, 1), big_y.reshape(big_x.size, 1)]
big_z = clf.decision_function(big_xy)

#
dec_errors = []
test_errors = []
for i in range(big_xy.shape[0]):
    dec_errors.append(objective(big_xy[i], clf, data, target='dec_val'))
    test_errors.append(objective(big_xy[i], clf, data, target='clferr'))
dec_errors = np.asarray(dec_errors).reshape(big_x.shape)
test_errors = np.asarray(test_errors).reshape(big_x.shape)
ax = plt.subplot(121)
ax.plot(data['train'][data['train_labels'] == 1, 0],
        data['train'][data['train_labels'] == 1, 1], 'ro')
ax.plot(data['train'][data['train_labels'] == -1, 0],
        data['train'][data['train_labels'] == -1, 1], 'bo')
ca = ax.contourf(big_x, big_y, dec_errors, 900, cmap='jet_r')
ax.contour(big_x, big_y, big_z.reshape(big_x.shape), [0])
plt.colorbar(ca, shrink=0.44)

# now random pick a sample and do the poisoning attack
pos_id = np.random.choice(np.where(data['train_labels'] == 1)[0], 1)
neg_id = np.random.choice(np.where(data['train_labels'] == -1)[0], 1)
x0 = np.r_[data['train'][pos_id], data['train'][neg_id]].mean(axis=0)
bds = list()
for j in np.arange(data['train'].shape[1]):
    bds.append((np.min(data['train'][:, j]), np.max(data['train'][:, j])))
grad_path = np.empty((0, 2))
grad_path = np.append(grad_path, [x0], 0)


def addpath(x):
    global grad_path
    grad_path = np.append(grad_path, [x], 0)

# TODO: learn more about this function to make it converge
print 'Gradient error: ', opt.check_grad(objective, grad_w, x0, clf, data)
res = opt.minimize(objective, x0.ravel(),
                   jac=grad_w,
                   args=(clf, data),
                   method='Newton-CG',
                   bounds=bds,
                   callback=addpath,
                   options={'disp':True, 'eps':1e-2, 'xtol':1e-12})
grad_path = np.asarray(grad_path)
ax.plot(grad_path[:, 0], grad_path[:, 1], 'ko-', lw=1, ms=2)
ax.plot(res.x[0], res.x[1], 'go', ms=4)
ax.plot(x0[0], x0[1], 'ko', ms=4)

# set ax properties
ax.set(title='Decision value')
utils.setAxSquare(ax)

##
bx = plt.subplot(122)
bx.plot(data['train'][data['train_labels'] == 1, 0],
        data['train'][data['train_labels'] == 1, 1], 'ro')
bx.plot(data['train'][data['train_labels'] == -1, 0],
        data['train'][data['train_labels'] == -1, 1], 'bo')
cb = bx.contourf(big_x, big_y, test_errors, 900)
bx.contour(big_x, big_y, big_z.reshape(big_x.shape), [0])
plt.colorbar(cb, shrink=0.44)

# set ax properties
bx.set(title='Classification error')
utils.setAxSquare(bx)

plt.tight_layout()
plt.show()

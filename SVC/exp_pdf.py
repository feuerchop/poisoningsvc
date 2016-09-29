'''
Created on Mar 24, 2015

@author: Xiao, Huang
'''

from sklearn.metrics import confusion_matrix as cfmat

from SVC.LibPoisonSVC import *


# load all data first
isLoad = False
if isLoad:
    print 'loading dataset ...'
    data = sio.loadmat('pdf_data/pdf100p10.mat')
    X = data['X']
    y = data['y'].ravel()
else:
    # sample 90% benign and 10% malicious
    print '... resampling dataset'
    data = sio.loadmat('pdf_data/norm_pdf_5k.mat')
    Xtr = data['X']
    ytr = data['y'].ravel()
    p,n = 0.4,0.6
    N = 100
    pid, nid = np.where(ytr==1)[0], np.where(ytr==-1)[0]
    x_pid, x_nid = np.random.choice(pid, np.floor(N*p)), np.random.choice(nid, np.floor(N*n))
    X = np.concatenate([Xtr[x_pid], Xtr[x_nid]], axis=0)
    y = np.concatenate([ytr[x_pid], ytr[x_nid]], axis=0)
    sio.savemat('pdf_data/pdf100p10.mat', mdict={'X':X, 'y':y})
    
print 'Training size: %d | benign samples: %d | malicious samples: %d ' % (X.shape[0], (y==-1).sum(), (y==1).sum())
clf = SVC(C=0.06, gamma=4)
clf.fit(X)
cmat = cfmat(y, clf.y[:y.size], labels=[1,-1])
print cmat
print '[Before attack] FNR = %.5f' % (float(cmat[0,1])/cmat[0].sum())
ntk = 20
x0 = X[np.random.choice(clf.sv_inx, ntk)]
bds = list()
for j in np.arange(X.shape[1]):
    bds.append((0, 1))
xc, fval, gradient_path, info = gradient_poison(x0, clf, X, bounds=bds, tol=1e-5, step_size=1)
cmat = cfmat(y, clf.y[:y.size], labels=[1,-1])
print cmat
print '[After attack] FNR = %.5f' % (float(cmat[0,1])/cmat[0].sum())


__author__='morgan'

import numpy as np
from sklearn.svm import OneClassSVM


class OCSVM(OneClassSVM):
    """
    One-class SVM class subclassed from OneClassSVM defined in sklearn
    Because the super predict function is not what we expected. We need
    to rewrite it to meet our needs.
    """

    # TODO: Add more properties to OCSVM
    def __init__(self, *args, **kwargs):
        super(OCSVM, self).__init__(*args, **kwargs)
        self.eps = 1e-4
        self.fval = None
        self.sv_ind = []
        self.bsv_ind = []

    def fit(self, *args, **kwargs):
        super(OCSVM, self).fit(*args, **kwargs)
        self.sv_ind = self.support_[self.dual_coef_.ravel() < 1-self.eps]
        self.bsv_ind = np.setdiff1d(self.support_, self.sv_ind)

    def predict_y(self,X):
        """
        :rtype : ndarray
        :param X: training data X
        :return: predicted labels, +1 for outlier, -1 for normal
        """
        dec=self.decision_function(X)
        # find the nearest BSV
        threshold = self.decision_function(X[self.sv_ind, :]).min()
        yc = dec.ravel()
        pos = yc < threshold
        neg = yc >= threshold
        yc[neg] = -1
        yc[pos] = 1
        return yc


if __name__=='__main__':
    # import other pkg
    import matplotlib.pylab as plt
    import scipy.io as sio
    from utils import getBoxbyX,setAxSquare
    from sklearn.metrics import accuracy_score as acc

    xx,yy=np.meshgrid(np.linspace(-5,5,500),np.linspace(-5,5,500))
    X=sio.loadmat('data/toy_sample.mat')['X']
    y=sio.loadmat('data/toy_sample.mat')['y'].ravel()
    m,d=X.shape
    Ax,Ay=getBoxbyX(X,grid=30)
    ax1 = plt.subplot(1,2,1)
    ax1.plot(X[y==-1,0], X[y==-1,1], 'bo')
    ax1.plot(X[y==1,0], X[y==1,1], 'ro')
    setAxSquare(ax1)
    all_points_x=np.c_[Ax.ravel(),Ay.ravel()]
    N=all_points_x.shape[0]
    err_fval=np.ones((N,1))

    # fit the classifier
    clf=OCSVM(nu=0.1, gamma=1)
    clf.fit(X)
    dist_all=clf.decision_function(all_points_x)
    labels=clf.predict_y(X)

    err=1-acc(y.ravel(), labels)
    ax2 = plt.subplot(1,2,2)
    ax2.set(title='Error: %.2f%%'%(100*err))
    ax2.plot(X[labels==1,0], X[labels==1,1], 'ro')
    ax2.plot(X[labels==-1,0], X[labels==-1,1], 'bo', ms=6)
    ax2.plot(X[clf.sv_ind,0],X[clf.sv_ind,1], 'o', mfc='none',ms=12)
    ax2.contour(Ax,Ay,dist_all.reshape(Ax.shape),levels=[0],colors='k',linestyles='solid')
    cb=plt.contourf(Ax,Ay,dist_all.reshape(Ax.shape), 10, cmap='bone', vmin=-5.8, vmax=1.2)
    setAxSquare(ax2)
    # plt.colorbar(cb,ax=ax2,format='%2.3f', aspect=12)
    plt.show()


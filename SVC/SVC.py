from utils import *
from sklearn import svm, datasets
from sklearn.metrics.pairwise import pairwise_kernels as kernel
from scipy import optimize as opt
from scipy import io as sio
from cvxopt import matrix as cvxmatrix
from cvxopt import solvers
import matplotlib as mpl

mpl.rc('lines', markersize=5)

#===============================================================================
# Test on one-class outliers, test on sensetic data(artificial data) 
# generate dataset like the one from SVC toolbox
# Exp.1 Synthetic dataset for outliers, gaussian ring high density and clusters around with low density the outer clusters are outliers
#===============================================================================

class SVC:
    def __init__(self, kernel='rbf', method='qp', \
                 C=1, gamma=0.5, coef0=1, degree=3, eps=1e-3, labeling=False, display=False):
            self.eps = eps
            self.method = method            # optimization methods, default use qp-solver
            self.kernel = kernel            # kernel string, default is rbf, only 'rbf', 'linear', 'poly' are supported
            self.alpha = None               # dual variables
            self.sv_ind = None              # indices of SVs
            self.bsv_ind = None             # indices of BSVs
            self.inside_ind = None          # indices of insides+SVs
            self.b = None                   # squared norm of center a
            self.sv = None                  # SVs + BSVs     
            self.sv_inx = None              # indices of SVs+BSVs
            self.nsv = None                 # number of SVs+BSVs
            self.r = None                   # radius of the hypersphere
            self.rid = None                 # Index of the SV for R
            self.K = None                   # cache of the kernel matrix on X
            self.C = C                      # regularizer coef, default 1, no outlier is allowed
            self.gamma = gamma              # param for rbf/poly kernel, default 1
            self.coef0 = coef0              # param for polynomial kernel, default 1
            self.degree = degree            # param for polynomial kernel, default 3
            self.fval = None                # objective function value after converge
            self.cluster_labels = None      # cluster labels
            self.y = None                   # class labels, +1 normal, -1 outlier
            self.labeling = labeling        # call labeling process, default False
            self.display = display          # show details of solver
            
    #===========================================================================
    # fit function: 
    # X: nxd data set
    #===========================================================================
    def fit(self, X):
        if self.display:
            solvers.options['show_progress'] = True
        else:
            solvers.options['show_progress'] = False
        
        n = X.shape[0]
        
        # Step 1: Optimizing the SVDD dual problem.....
        K = kernel(X, metric=self.kernel, n_jobs=1, \
                   filter_params=True, gamma=self.gamma, \
                   coef0=self.coef0, degree=self.degree)
        q = cvxmatrix(-K.diagonal(), tc='d')
        
        if self.method is 'qp':
            P = cvxmatrix(2*K, tc='d')
            G = cvxmatrix(np.vstack((-np.eye(n), np.eye(n))), tc='d')                   # lhs box constraints
            h = cvxmatrix(np.concatenate((np.zeros(n), self.C*np.ones(n))), tc='d')     # rhs box constraints
    
            # optimize using cvx solver
            sol = solvers.qp(P,q,G,h,initvals=cvxmatrix(np.zeros(n), tc='d'))
            
        if self.method is 'smo':
            #TODO: apply SMO algorithm    
            alpha = None
            
        # setup SVC model
        alpha = np.asarray(sol['x'])
        inx = np.where(alpha > self.eps)[0]  
        self.sv_inx = inx       
        self.alpha = alpha[inx]
        self.nsv = inx.size
        self.sv_ind = np.where((alpha > self.eps) & (alpha < self.C-self.eps))[0]
        self.bsv_ind= np.where(alpha >= self.C-self.eps)[0]
        self.inside_ind = np.where(alpha < self.C-self.eps)[0]
        k_inx = K[inx[:,None], inx]                 # submatrix of K(sv+bsv, sv+bsv)
        k_sv = K[self.sv_ind[:,None], self.sv_ind]  # submatrix of K(sv,sv)
        k_bsv = K[self.bsv_ind[:,None], self.bsv_ind]  # submatrix of K(bsv,bsv)
        # 2-norm of center a^2
        self.b = self.alpha.reshape(1,self.nsv).dot(k_inx).reshape(1,self.nsv).dot(self.alpha.reshape(self.nsv,1))
        #including both of SV and BSV (bounded support vectors)
        self.sv= X[inx, :]
        d = k_sv.diagonal() - 2*self.alpha.reshape(1,self.nsv).dot(K[inx[:,None], self.sv_ind]) + self.b * np.ones(self.sv_ind.size)
        self.r = d.max()
        self.rid = self.sv_ind[np.argmax(d.ravel())]
        d_bsv = k_bsv.diagonal() - 2*self.alpha.reshape(1,self.nsv).dot(K[inx[:,None], self.bsv_ind]) + self.b * np.ones(self.bsv_ind.size)
        self.fval = self.r+self.C*(d_bsv - self.r).sum()
        self.K = K
        self.y = -1*np.ones(n)
        self.y[self.bsv_ind] = 1
        if self.labeling:
            #Step 2: Labeling cluster index by using CG
            self.predict(X)
    
    # predict the cluster labels using CG
    def predict(self, X):
        N = X.shape[0]
        adjacent = self.FindAdjMatrix(X[self.inside_ind, :])
        clusters = self.FindConnectedComponents(adjacent)
        self.cluster_labels = np.zeros(N)
        self.cluster_labels[self.inside_ind] = np.double(clusters)
        return
   
    # predict labels on Xtt 
    def predict_y(self, Xtt):
        dist_tt = self.kdist2(Xtt)
        y = -1*np.ones(Xtt.shape[0])
        y[dist_tt.ravel() > self.r] = 1
        return y
        
    # Description
    #    The Adjacency matrix between pairs of points whose images lie in
    #    or on the sphere in feature space. 
    #    (i.e. points that belongs to one of the clusters in the data space)
    #
    #    given a pair of data points that belong to different clusters,
    #    any path that connects them must exit from the sphere in feature
    #    space. Such a path contains a line segment of points y, such that:
    #    kdist2(y,model)>model.r.
    #    Checking the line segment is implemented by sampling a number of 
    #   points (10 points).
    #    
    #    BSVs are unclassfied by this procedure, since their feature space 
    #    images lie outside the enclosing sphere.( adjcent(bsv,others)=-1 )
    def FindAdjMatrix(self, X):
        N = X.shape[0]
        adjacent = np.zeros((N, N))
        R = self.r + self.eps  # Squared radius of the minimal enclosing ball
        for i in xrange(0, N-1):
            for j in xrange(0, N-1):
                # if the j is adjacent to i - then all j adjacent's are also adjacent to i
                if j<i:
                    if adjacent[i, j] == 1:
                        adjacent[i,:] = adjacent[i,:] | adjacent[j,:]
                else:
                    # if adajecancy already found - no point in checking again
                    if adjacent[i, j] != 1:
                        # goes over 10 points in the interval between these 2 Sample points
                        # unless a point on the path exits the shpere - the points are adjacnet
                        adj_flag = 1                        
                        for interval in np.arange(0,1,0.1):
                            z = X[:,i] + interval * (X[:,j] - X[:,i])
                            # calculates the sub-point distance from the sphere's center 
                            d = self.kdist2(z)
                            if d > R:
                                adj_flag = 0
                                break
                        if adj_flag == 1:
                            adjacent[i,j] = 1
                            adjacent[j,i] = 1
        return adjacent

    # Compute the distances of X with the ball center
    # Output:
    #  d [num_data x 1] Squared distance between vectors in the feature space and the center hyperball
    def kdist2(self, X):
        n = X.shape[0]
        K = kernel(X, metric=self.kernel, filter_params=True, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        f = K.diagonal()
        K_sv = kernel(X, self.sv, metric=self.kernel, filter_params=True, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        d = f - 2*self.alpha.reshape(1,self.nsv).dot(K_sv.T) + self.b * np.ones(n)
        return d
    
    #    clusters_assignments - label each point with its cluster assignement.
    #Finds Connected Components in the graph represented by  
    #the adjacency matrix, each component represents a cluster.
    def FindConnectedComponents(self, adjacent):
        N = adjacent.shape[0]
        clusters_assignments = np.zeros(N)
        cluster_index = 0
        done = 0
        while done != 1:
            root = 1
            while clusters_assignments[root] != 0:
                root = root + 1
                if root > N: #all nodes are clustered
                    done = 1
                    break
            if done != 1: #an unclustered node was found - start DFS
                cluster_index = cluster_index + 1
                stack = np.zeros(N)
                stack_index = 0
                while stack_index != 0:
                    node = stack[stack_index]
                    stack_index = stack_index - 1
                    clusters_assignments[node] = cluster_index
                    for i in xrange(0, N-1):
                        #check that this node is a neighbor and not clustered yet
                        if (adjacent[node,i] == 1 & clusters_assignments[i] == 0 & i != node):
                            stack_index = stack_index + 1
                            stack[stack_index] = i
        return clusters_assignments



# main function to test the SVC Model
if __name__ == '__main__':
    clf = SVC(C=0.2, gamma=1)
    #generate data
    info = sio.loadmat('ring.mat')
    X = info['train_input']
    y = info['train_output']    
    Ax,Ay = getBoxbyX(X,grid=30)
    all_points_x = np.concatenate( (Ax.ravel().reshape(Ax.size, 1), 
                              Ay.ravel().reshape(Ay.size, 1)), 1)
    #build SVC model
    clf.fit(X)
    # distance to center a
    dist_all = clf.kdist2(all_points_x)
    #plot 
    outliers = X[clf.bsv_ind, :]
    plt.plot(X[clf.inside_ind, 0], X[clf.inside_ind, 1], 'bo')
    plt.plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'bo')
    plt.plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'o', markerfacecolor='None', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    plt.plot(X[clf.bsv_ind, 0], X[clf.bsv_ind, 1], 'ko')
    plt.contourf(Ax,Ay,dist_all.reshape(Ax.shape), 100, cmap='bone_r', alpha=0.4)
    plt.contour(Ax,Ay,dist_all.reshape(Ax.shape), levels=[clf.r], colors='k', linestyles='solid', linewidth=1.2)
    xlim0, xlim1 = plt.gca().get_xlim()    
    ylim0, ylim1 = plt.gca().get_ylim()    
    plt.gca().set_aspect((xlim1-xlim0)/(ylim1-ylim0))
    plt.show()
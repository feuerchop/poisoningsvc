'''
Created on Mar 13, 2015

@author: Xiao, Huang
'''
from scipy import io as sio
from numpy.linalg import norm
from scipy.linalg import solve
import sklearn.preprocessing as prep
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc

from SVC.SVC import *
from utils import *


#===============================================================================
# xval the kernel parameters (gamma/coef0/...) for SVC
# Inputs: Xtr 
# Return: SVC model
#===============================================================================
def xval_models(Xtr):
    pass

#===============================================================================
# we update multipile points in TR dataset, retrain the classifier and return 
# a contaminated classifier under attack
# Input: xc:  a mxd ndarray for attack points
#        clf: the classifier 
#        Xtr: the training dataset already containing the attack points
#        xcid: 1d array of indices of the attack points in Xtr
#        err_type: the measured error type, 
#                  default is the objective function a value
# Return: The negative of the measured error
#===============================================================================
def updatePointsAtk(xc, clf, Xtr, ytr=None, Xtt=None, ytt=None, err_type='loss'):
    d = Xtr.shape[1]
    m = xc.size/d
    xc = xc.reshape(m,d)
#     if xcid.size != m:
#         print 'Attack points indices do not match xc, exit!'
#         return None
    X0 = np.concatenate([Xtr, xc], axis=0)
    # TODO: update SVC instead of retrain with xc
    clf.fit(X0)        # <---- this is just a lazy update
    # maximize the obj value (generalization error)   
    if err_type is 'fval':
        # objective values on untainted dataset Dtr
        return 1*(clf.fval - clf.C*(clf.kdist2(xc) - clf.r).sum())
    elif err_type is 'r':
        # the squared radius 
        return 1*clf.r
    elif err_type is 'xi':
        return 1*(clf.fval - clf.C*(clf.kdist2(xc) - clf.r).sum() - clf.r)
    elif err_type is 'f1':  
        if ytr is not None:
            return f1_score(ytr, clf.y[:ytr.size])
        else:
            print 'You need give the true labels!'
            return None
    elif err_type is 'acc':  
        if Xtt is not None and ytt is not None:
            y_clf = clf.predict_y(Xtt)
            return acc(ytt, y_clf)
        else:
            print 'You need give the test dataset!'
            return None
    elif err_type is 'loss':
        # min(\sum_xi - R^2) means let less samples lie out meanwhile maximize the ball
        # note: xc are excluded
        sum_xi_c = (clf.kdist2(xc) - clf.r).sum()
        return (clf.fval - clf.r)/clf.C - sum_xi_c - clf.r
    elif err_type is 'fn':
        if ytr is not None:
            pid = np.where(ytr==1)[0]
            return 0.5*((ytr[pid] - clf.y[pid]).sum())/pid.size
        else:
            print 'You need give the true labels!'
            return None
    elif err_type is 'fp':
        if ytr is not None:
            nid = np.where(ytr==-1)[0]
            return 0.5*((clf.y[nid] - ytr[nid]).sum())/nid.size
        else:
            print 'You need give the true labels!'
            return None    
            
    
#===========================================================================
# gradient of objective W wrt. x_c
# compute the gradient of xc in X given the indices as xcid
# classifier must be trained on X first! We avoid retraining gradient while
# computing the gradient!
# Inputs:    xc, 1xd ndarrady for single attack point
#            clf, classifier on current iteration
#            X, nxd dataset containing attack points already
#            xcid, indices of xc, default is size of X, last index of X
#===========================================================================
def grad_w(xc, clf, X):   
#     # basic condition check
#     if (clf.K is None) or (clf.K.shape[0] != X.shape[0]):
#         print 'Either the classifier is not trained at all or not updated on X, exit!'
#         return None 
#     
#     if ~np.equal(xc,X[xcid]).all():
#         print 'Index of the attack point does not match the data, exit!'
#         return None
    # Initialize
    n,d = X.shape
#     xc_idx = xcid
    xc = xc.reshape(1,d)
    X0 = np.concatenate([X,xc], axis=0)         # <--- xc is always put in the end with index n
    # compute kernel gradients of xc
    if clf.kernel is 'linear':
        grad_kc = X0
        grad_kc[n,:] = 2*grad_kc[n,:]
    elif clf.kernel is 'poly':
        grad_kc = X0*np.repeat( (clf.degree*(X0.dot(xc.T)+clf.coef0)**(clf.degree-1)).reshape(n+1,1), d, axis=1)
        grad_kc[n,:] = 2*grad_kc[n,:]
    elif clf.kernel is 'rbf':    
        xc_mat = np.repeat(xc, n+1, axis=0)
        grad_kc = (X0-xc_mat)*np.repeat(2*clf.gamma*np.exp(-1*clf.gamma*norm(X0-xc_mat,2,axis=1)**2).reshape(n+1,1), d, axis=1)
        grad_kc[n,:] = 2*grad_kc[n,:]
    
    # fit SVC on X0
    clf.fit(X0)
    
    # vector of gradient alpha
    v = np.zeros((n+1,d))
    if n in clf.sv_inx:
        xc_alpha_idx = np.where(clf.sv_inx==n)[0]
        alpha_c = clf.alpha[xc_alpha_idx]
#         if n in clf.sv_ind:
#             print 'xc in SVs, alpha_c=',alpha_c
#         else:
#             print 'xc in BSVs, alpha_c=',alpha_c
    else:
        # attack point inside
        alpha_c = 0  
#         print 'xc inside'  
    K_sv = clf.K[clf.sv_ind[:,None], clf.sv_ind]
    lhs = np.repeat(K_sv[0].reshape(1,K_sv[0].size), clf.sv_ind.size-1, axis=0) - K_sv[1:] 
    lhs = np.vstack((lhs, np.ones((1, clf.sv_ind.size))))
    # numerical correction
    lhs=lhs+1e-6*np.eye(lhs.shape[0])
    grad_kc_sv = grad_kc[clf.sv_ind, :]
    rhs = alpha_c*(grad_kc_sv[1:] - \
                         np.repeat(grad_kc_sv[0].reshape(1,grad_kc_sv[0].size), clf.sv_ind.size-1, axis=0))
#     if n in clf.sv_ind:
# #         xc_sv_idx = np.where(clf.sv_ind==n)[0]
#         rhs[-1] = clf.alpha.reshape(1, clf.nsv).dot(grad_kc[clf.sv_inx,:]) - alpha_c*grad_kc_sv[0] - 0.5*alpha_c*grad_kc[n]
    rhs = np.vstack((rhs, np.zeros((1, d))))
    
    # solve the linear system by lstsq
#     vs, residuals, rank, s = lstsq(lhs, rhs)
    vs = solve(lhs, rhs)
    # correct the solution according to KKT(1)
#     vs[0] = -vs[1:].sum(axis=0)
#         print 'residuals: ', residuals
#         print 'rank: %d lhs_rows: %d ' % (rank, clf.sv_ind.size-1)
    v[clf.sv_ind] = vs
#     print 'sum of v: ', v.sum(axis=0)
    # grad_R^2
    # randomly choose a sv/x_c
#     s_idx = np.random.choice(np.setdiff1d(clf.sv_ind, n), 1)
    s_idx = clf.rid       # <-- use the first SV as R2 as in previous computation
    grad_R2 = -2*(clf.K[s_idx, :].dot(v) - clf.alpha.reshape(1, clf.nsv).dot(clf.K[clf.sv_inx, :]).reshape(1,n+1).dot(v) + \
                     alpha_c*grad_kc[s_idx] - alpha_c*clf.alpha.reshape(1, clf.nsv).dot(grad_kc[clf.sv_inx,:]) + \
                     0.5*alpha_c**2*grad_kc[n])
    grad_R2 = grad_R2.ravel()
    
    # grad_\sum_\xi
#     n_bsv = clf.bsv_ind.size
#     if n in clf.bsv_ind:
#         indicator = 1
#     else:
#         indicator = 0
#     grad_sum_xi = -2*((np.ones((1,n_bsv)).dot(clf.K[clf.bsv_ind,:]) - \
#                    n_bsv*clf.alpha.reshape(1,clf.nsv).dot(clf.K[clf.sv_inx,:])).reshape(1,n).dot(v) + \
#                    alpha_c*np.ones((1,n_bsv)).dot(grad_kc[clf.bsv_ind,:]) - \
#                    n_bsv*alpha_c*clf.alpha.reshape(1, clf.nsv).dot(grad_kc[clf.sv_inx, :])) - \
#                    n_bsv*alpha_c**2*grad_kc[n] + indicator*(2*alpha_c*clf.alpha.reshape(1, clf.nsv).dot(grad_kc[clf.sv_inx, :]) + \
#                                                                  (2*alpha_c+1)*grad_kc[n]) - n_bsv*grad_R2
    bsv_ind_noxc = np.setdiff1d(clf.bsv_ind, n)
    bsv_ind_num =  bsv_ind_noxc.size
    # grad_sum_xi without considering xc
    grad_sum_xi = -2*( ( np.ones((1,bsv_ind_num)).dot(clf.K[bsv_ind_noxc,:]) - \
                    bsv_ind_num*clf.alpha.reshape(1,clf.nsv).dot(clf.K[clf.sv_inx,:]) ).reshape(1,n+1).dot(v) + \
                    alpha_c*np.ones((1,bsv_ind_num)).dot(grad_kc[bsv_ind_noxc,:]) - \
                    bsv_ind_num*alpha_c*clf.alpha.reshape(1, clf.nsv).dot(grad_kc[clf.sv_inx]) - \
                    0.5*bsv_ind_num*alpha_c**2*grad_kc[n]) - bsv_ind_num*grad_R2
    grad_sum_xi = grad_sum_xi.ravel()
#     return -1*(grad_R2+clf.C*grad_sum_xi)   
#     return -1*(clf.C*grad_sum_xi)   
#     return -1*grad_R2
    return grad_sum_xi - grad_R2
 
#===============================================================================
# Gradient descent based attack ,input initial attack points, we find the final 
# malicious points and its gradient path, as long as the objective function value fval
# Input: x0 is a m*d ndarray for the initial attack points
#        clf is the classifier to be attacked
#        Xtr is the n*d ndarray for tranining dataset to be contaminated
#        bounds is a list of tuples (min, max) for the range of each feature
#        max_iter: max. iteration for optimization
#        tol: tolerance for optimization
#        step_size: step size for gradient
#        display: show details while optimizing
# Return: x is a m*d ndarray of the optimal attack points
#         fval is the optimal objective function value
#         gradient_path is a (?,d) ndarray containing attack points along the gradient path
#         info is a dict containing the optimization information
#===============================================================================

def gradient_poison(x0, clf, Xtr, bounds=[], max_iter=1000, tol=1e-5, step_size=1, display=True):
    beta = 0.9
    status = True
    # copy the initial x0 to make sure it will never be changed
    x_init = x0.copy()
    m,d = Xtr.shape
    n_atk = x0.shape[0]
    gradient_path = np.empty(shape=[0, n_atk, d])
    fval_path = list()
    
    # set the step_size according to the attack points size
    # the more attack points we have, the more difficult it can converge
    step_sizes = beta**np.floor(n_atk/5)*step_size*np.ones(n_atk)

    # fit the classifier by adding initial attacks
#     X0 = np.concatenate([Xtr, x_init], axis=0)
#     xc_indices = m+np.arange(0,n_atk)
    fval = updatePointsAtk(x_init, clf, Xtr)         # <--- first trained on contaminated X
    fval_path.append(fval)
    
    if display:
        print 'Fval on initial attacks: %f', fval
    for iter in range(max_iter):
        if display:
            print '[%d] Iteration ... on OCSVM ...' % (iter)
        # put x into gradient path
        gradient_path = np.append(gradient_path, [x_init], 0)
        
        # take gradient point by point iteratively
        for k in range(n_atk):
            # append other points on Xtr
            xtr_new = np.append(Xtr, x_init[0:k].reshape(k,d), axis=0)
            xtr_new = np.append(xtr_new, x_init[k+1:].reshape(n_atk-k-1, d), axis=0)
            # take the gradient on xc point by point
            grad = grad_w(x_init[k], clf, xtr_new)
            if grad is None:
                print 'error happens in gradient computation!'
                return None
            # update xc ???
            x_init[k] = x_init[k] - step_sizes[k]*grad
            
            # bound the attack point
            for j in range(d):
                if x_init[k,j] < bounds[j][0]:
                    x_init[k,j] = bounds[j][0]
                if x_init[k,j] > bounds[j][1]:
                    x_init[k,j] = bounds[j][1]
            
            # update X and classifier
            fval_new = updatePointsAtk(x_init[k], clf, xtr_new)   
            if display:
                print '____Gradient on %d-th point, fval=%f, |proj g| = %f, step_size = %f' % (k+1, fval_new, norm(grad, 2), step_sizes[k])
                
        # after moving all the attack points, append the new fval for this iteration    
        fval_path.append(fval_new)    
        delta_f = fval_new-fval
        if delta_f > tol:
            # step size is too big
            step_sizes = step_sizes*beta
        if display:
            print '____[%d points shifted] fval=%f | delta_f=%f' % (n_atk, fval_new, delta_f)
        if np.abs(delta_f) < tol:
            print '\tCONVERGED AT ITERATION %d | fval=%f' % (iter, fval)
            info = {'fvals':np.array(fval_path), 'iter':iter, 'status':status}
            return (x_init, fval, gradient_path, info)
        else:
            fval = fval_new
            
    status = False
    print '\tNOTCONVERGE: Maximal iteration reached! fval=%f, delta_f=%f' % (fval, delta_f)
    gradient_path = np.append(gradient_path, [x_init], 0)
    info = {'fvals':np.array(fval_path), 'iter':iter, 'status':status}
    return (x_init, fval, gradient_path, info)    
        
        
if __name__ == '__main__':    
    # check gradient
    info = sio.loadmat('ring.mat')
    X = info['train_input']
    y = info['train_output']
#     X,y=genMulti2gauss(n_samples_each=50, n_features=10, max_mean=10, min_mean=1, var=1)
    X = prep.scale(X)
    n,d = X.shape
    clf = SVC(C=0.1,gamma=1)
    clf.fit(X)
    xc = X[np.random.choice(clf.sv_inx, 1)]
#     X0 = np.concatenate([X, xc.reshape(1,d)], axis=0)
#     clf.fit(X0)
    err = opt.check_grad(updatePointsAtk, grad_w, xc, clf, X)
    print 'gradient error on xc = ',err
    
    
    
    
    
'''
Created on Mar 13, 2015

@author: Xiao, Huang
'''
from SVC.LibPoisonSVC import *
# 
 
# gradient_path = np.empty((0,2))
# def add_gpath(x):
#     global gradient_path
#     gradient_path = np.append(gradient_path, [x], 0)

if __name__ == '__main__':
    # Subplot properties
    fig = getSquareSubplots(rows=1, cols=2, 
                  fig_w=14, fig_h=5,
                  pad_x=.05, pad_y=.06, 
                  gutter_x=.12, gutter_y=.08)
    axs = fig.get_axes()
    isLoad = False
    ## load data
#     info = sio.loadmat('ring.mat')
#     X = info['train_input']
#     y = info['train_output']
    X,y = gen_2gauss([15,50], mu=np.array([[1.5,0],[0,1.5]]), c1=[[0.1,0],[0,0.1]], c2=[[0.1,0],[0,0.1]])
#     X,y=gen_svc_data([0,0], 2.5, cluster_sizes=[100,10,10,10])
    X = prep.scale(X)
    m,d = X.shape
    Ax,Ay = getBoxbyX(X,grid=30)
#     Ax,Ay = np.meshgrid(np.linspace(-2.3, 2.3, grid), np.linspace(-2.3, 2.3, grid))
    all_points_x = np.concatenate( (Ax.ravel().reshape(Ax.size, 1), 
                              Ay.ravel().reshape(Ay.size, 1)), 1)
    N  = all_points_x.shape[0]
    err_fval = np.ones((N,1)) 
    clf = SVC(C=0.15, gamma=2)
    clf.fit(X)
    R2 = clf.r          # <--- squared radius without attack
    # distance to center a
    dist_all = clf.kdist2(all_points_x)
        
    insiders = X[clf.inside_ind, :]
    svs = X[clf.sv_ind, :]
    bsvs = X[clf.bsv_ind, :]
    axs[0].set_title('Original samples')
    axs[0].plot(insiders[:, 0], insiders[:, 1], 'o', ms=5, markerfacecolor='none')
    axs[0].plot(svs[:, 0], svs[:, 1], 'o', markerfacecolor='None', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[0].plot(bsvs[:, 0], bsvs[:, 1], 'ko', ms=5)
    ca=axs[0].contourf(Ax,Ay,dist_all.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    axs[0].contour(Ax,Ay,dist_all.reshape(Ax.shape), levels=[R2], colors='k', linestyles='solid', linewidth=1.2)
    setAxSquare(axs[0]) 
    
    # randomly choose sv points to start attack
    n_atk = 1
    x0 = X[np.random.choice(clf.sv_inx, n_atk)]  
#     x0 = np.array([[1,1], [1.5,0], [-1,0], [-0.5,-1.8]])
    bds = list()
    for j in np.arange(X.shape[1]):
        bds.append((np.min(X[:, j]), np.max(X[:, j])))
    
    if isLoad:
        err_fval = sio.loadmat('demo_err')['err']   
    else:
        err_fval=Parallel(n_jobs=1,max_nbytes=None) \
                        (delayed(updatePointsAtk)(all_points_x[i], clf, X, ytr=y, err_type='loss') \
                         for i in range(N))
        err_fval = np.asarray(err_fval, dtype='double')   
        sio.savemat('demo_err', mdict={'err':err_fval})     
        
    # numerical gradient descent
    
#     xc,f,info=opt.fmin_l_bfgs_b(updatePointsAtk, x0, args= (clf, X), approx_grad=True, bounds=bds, callback=add_gpath)
    xc, fval, gradient_path, info = gradient_poison(x0, clf, X, bounds=bds, tol=1e-5, step_size=0.1)      
    Xcc = np.concatenate([X,xc],axis=0)                   
    # plotting error surface and boundary after attack 
    axs[1].set_title('Objective Loss')
    axs[1].plot(Xcc[clf.inside_ind, 0], Xcc[clf.inside_ind, 1], 'o', ms=5, markerfacecolor='none')
    axs[1].plot(Xcc[clf.sv_ind, 0], Xcc[clf.sv_ind, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[1].plot(Xcc[clf.bsv_ind, 0], Xcc[clf.bsv_ind, 1], 'ko', ms=5)
    # plot boundary without attack
    dist_cc = clf.kdist2(all_points_x)
    axs[1].contour(Ax,Ay,dist_cc.reshape(Ax.shape), levels=[clf.r], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    cb=axs[1].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    # plot gradient path
    for atk in range(n_atk):
        axs[1].plot(gradient_path[:,atk,0],gradient_path[:,atk,1],'k-',lw=2)
        axs[1].plot(xc[atk, 0],xc[atk, 1],'go',ms=10,markeredgecolor='g', markerfacecolor='none', markeredgewidth=2)
    
    plot_bounds(bds, axs[1])
    setAxSquare(axs[1])  
    # add colorbar
    cbar_ax = fig.add_axes([.375, .06, .025, 0.88])
    fig.colorbar(ca, cax=cbar_ax, format='%2.3f')
    cbar_bx = fig.add_axes([.81, .06, .025, 0.88])
    fig.colorbar(cb, cax=cbar_bx, format='%2.3f')
    plt.show()
    
        
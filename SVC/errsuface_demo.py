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
# mpl.rc('xtick', labelsize=18)

if __name__ == '__main__':
    isParallel = 1
    # Subplot properties
    fig = getSquareSubplots(rows=3, cols=3, 
                  fig_w=10.5, fig_h=9.5,
                  pad_x=.05, pad_y=.06, 
                  gutter_x=.1, gutter_y=.07)
    axs = fig.get_axes()
    isLoad = False
    ## load data
#     info = sio.loadmat('ring.mat')
#     X = info['train_input']
#     y = info['train_output']
    X,y = gen_2gauss([10,50], mu=np.array([[1.5,0],[0,1.5]]), c1=[[0.1,0],[0,0.1]], c2=[[0.1,0],[0,0.1]])
    Xt,yt = gen_2gauss([10,50], mu=np.array([[1.5,0],[0,1.5]]), c1=[[0.1,0],[0,0.1]], c2=[[0.1,0],[0,0.1]])
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
    insider = clf.inside_ind
    svs = clf.sv_ind
    bsvs = clf.bsv_ind
    R = clf.r
    # distance to center a
    dist_a1 = clf.kdist2(all_points_x)
    # plot original data
    axs[0].set_title('Original SVC')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].plot(X[clf.inside_ind, 0], X[clf.inside_ind, 1], 'o', ms=5, markerfacecolor='none')
    axs[0].plot(X[clf.sv_ind, 0], X[clf.sv_ind, 1], 'o', markerfacecolor='None', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[0].plot(X[clf.bsv_ind, 0], X[clf.bsv_ind, 1], 'ko', ms=5)
    ca0=axs[0].contourf(Ax,Ay,dist_a1.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    axs[0].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    setAxSquare(axs[0]) 
    
    # error: R^2
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, err_type='r') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[1].set_title(r'$R^2$')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[1].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[1].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[1].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca1=axs[1].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    setAxSquare(axs[1])  
    
    # error: sum_xi
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, err_type='xi') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[2].set_title(r'$C\sum_j\xi_j, x_j\neq x_c$')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[2].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[2].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[2].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca2=axs[2].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # error: r^2+sum_xi
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, err_type='fval') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[3].set_title(r'$R^2+C\sum_j\xi_j, x_j\neq x_c$')
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    axs[3].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[3].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[3].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[3].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca3=axs[3].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # error: sum_xi - R^2
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, err_type='loss') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[4].set_title(r'$\sum_j\xi_j-R^2, x_j\neq x_c$')
    axs[4].set_xticks([])
    axs[4].set_yticks([])
    axs[4].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[4].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[4].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[4].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca4=axs[4].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    
    # error: accuracy on test set
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, ytr=y, Xtt=Xt, ytt=yt, err_type='acc') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[5].set_title(r'Accuracy')
    axs[5].set_xticks([])
    axs[5].set_yticks([])
    axs[5].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[5].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[5].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[5].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca5=axs[5].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # error: FN
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, ytr=y, err_type='fn') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[6].set_title(r'False negative rate')
    axs[6].set_xticks([])
    axs[6].set_yticks([])
    axs[6].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[6].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[6].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[6].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca6=axs[6].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # error: FP
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, ytr=y, err_type='fp') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[7].set_title(r'False positive rate')
    axs[7].set_xticks([])
    axs[7].set_yticks([])
    axs[7].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[7].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[7].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[7].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca7=axs[7].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # error: f1
    err_fval=Parallel(n_jobs=isParallel,max_nbytes=None) \
                    (delayed(updatePointsAtk)(all_points_x[i], clf, X, ytr=y, err_type='f1') \
                     for i in range(N))
    err_fval = np.asarray(err_fval, dtype='double')       
    # plotting error surface and boundary after attack 
    axs[8].set_title(r'F1 score')
    axs[8].set_xticks([])
    axs[8].set_yticks([])
    axs[8].plot(X[insider, 0], X[insider, 1], 'o', ms=5, markerfacecolor='none')
    axs[8].plot(X[svs, 0], X[svs, 1], 'o', markerfacecolor='none', ms=12, markeredgewidth=1.5, markeredgecolor='k')
    axs[8].plot(X[bsvs, 0], X[bsvs, 1], 'ko', ms=5)
    axs[8].contour(Ax,Ay,dist_a1.reshape(Ax.shape), levels=[R], colors='k', linestyles='solid', linewidth=1.2)
    # plot error surface
    ca8=axs[8].contourf(Ax,Ay,err_fval.reshape(Ax.shape), 100, cmap='jet', alpha=0.3)
    
    # add colorbar
    cbar_a0 = fig.add_axes([0.05+1*0.24, .94-0.245, .02, 0.245])
    fig.colorbar(ca0, cax=cbar_a0, format='%2.3f')
    cbar_a1 = fig.add_axes([0.05+2*0.24+0.08, .94-0.245, .02, 0.245])
    fig.colorbar(ca1, cax=cbar_a1, format='%2.3f')
    cbar_a2 = fig.add_axes([0.05+3*0.24+0.16, .94-0.245, .02, 0.245])
    fig.colorbar(ca2, cax=cbar_a2, format='%2.3f')
    cbar_a3 = fig.add_axes([0.05+1*0.24, .94-2*0.245-0.072, .02, 0.245])
    fig.colorbar(ca3, cax=cbar_a3, format='%2.3f')
    cbar_a4 = fig.add_axes([0.05+2*0.24+0.08, .94-2*0.245-0.072, .02, 0.245])
    fig.colorbar(ca4, cax=cbar_a4, format='%2.3f')
    cbar_a5 = fig.add_axes([0.05+3*0.24+0.16, .94-2*0.245-0.072, .02, 0.245])
    fig.colorbar(ca5, cax=cbar_a5, format='%2.3f')
    cbar_a6 = fig.add_axes([0.05+1*0.24, .94-3*0.245-2*0.072, .02, 0.245])
    fig.colorbar(ca6, cax=cbar_a6, format='%2.3f')
    cbar_a7 = fig.add_axes([0.05+2*0.24+0.08, .94-3*0.245-2*0.072, .02, 0.245])
    fig.colorbar(ca7, cax=cbar_a7, format='%2.3f')
    cbar_a8 = fig.add_axes([0.05+3*0.24+0.16, .94-3*0.245-2*0.072, .02, 0.245])
    fig.colorbar(ca8, cax=cbar_a8, format='%2.3f')
   
    plt.show()
    
        
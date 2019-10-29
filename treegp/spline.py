import numpy as np
import pylab as plt
import cPickle
from saunerie import bspline

def plot_correlation_function(interp, save=False, rep='',
                              specific_name_kernel='VK', NAME='du', exp='0', other_fit=None):
    
    if other_fit is None: 
        FIT = interp._2pcf_fit
    else:
        FIT = other_fit

    EXT = [np.min(interp._2pcf_dist[:,0]/60.), np.max(interp._2pcf_dist[:,0]/60.),
           np.min(interp._2pcf_dist[:,1]/60.), np.max(interp._2pcf_dist[:,1]/60.)]
    CM = plt.cm.seismic

    MAX = np.max(interp._2pcf)
    N = int(np.sqrt(len(interp._2pcf)))
    plt.figure(figsize=(14,5) ,frameon=False)
    plt.gca().patch.set_alpha(0)
    plt.subplots_adjust(wspace=0.5,left=0.07,right=0.95, bottom=0.15,top=0.85)
    plt.subplot(1,3,1)
    plt.imshow(interp._2pcf.reshape(N,N), extent=EXT, interpolation='nearest', origin='lower',
               vmin=-MAX, vmax=MAX, cmap=CM)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\xi$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)
    plt.title('Measured 2-PCF',fontsize=16)

    plt.subplot(1,3,2)
    plt.imshow(FIT.reshape(N,N), extent=EXT, interpolation='nearest',
               origin='lower',vmin=-MAX,vmax=MAX, cmap=CM)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\xi\'$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)

    var = return_var_map(interp._2pcf_weight, interp._2pcf)
    cm_residual = plt.matplotlib.cm.get_cmap('RdBu',10)
    Res = interp._2pcf[interp._2pcf_mask] - FIT[interp._2pcf_mask]
    #chi2 = Res.dot(interp._2pcf_weight).dot(Res)
    dof = np.sum(interp._2pcf_mask) #- 4.

    pull = (interp._2pcf.reshape(N,N) - FIT.reshape(N,N)) / np.sqrt(var)
    chi2 = np.sum((interp._2pcf.reshape(N,N) - FIT.reshape(N,N))**2 / var)

    plt.title('Fitted 2-PCF' ,fontsize=16)

    plt.subplot(1,3,3)

    plt.imshow(pull, extent=EXT, interpolation='nearest', origin='lower', vmin=-5., vmax=+5., cmap=cm_residual)
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    cbar.set_label('$\\frac{\\xi-\\xi\'}{\sigma_{\\xi}}$',fontsize=20)
    plt.xlabel('$\\theta_X$ (arcmin)',fontsize=20)
    plt.ylabel('$\\theta_Y$ (arcmin)',fontsize=20)
    plt.title('Pull',fontsize=16)
    plt.suptitle(NAME+' anisotropy 2-PCF (%s)  $\chi^2$ / dof = %.3f'%((specific_name_kernel, chi2/dof)), 
                 fontsize=16)

def return_var_map(weight, xi):
    N = int(np.sqrt(len(xi)))
    var = np.diag(np.linalg.inv(weight))
    VAR = np.zeros(N*N)
    I = 0
    for i in range(N*N):
        if xi[i] !=0:
            VAR[i] = var[I]
            I+=1
        if I == len(var):
            break
    VAR = VAR.reshape(N,N) + np.flipud(np.fliplr(VAR.reshape(N,N)))
    if N%2 == 1:
        VAR[N/2, N/2] /= 2.
    return VAR

class spline2D_hyp_search(object):

    def __init__(self, x, y, z, weight=None, order=4, nx=11, ny=11, limit=980.):

        # 2-point correlation functions, output 
        # of TreeCorr using 2D fit. 
        self.x = x
        self.y = y
        self.z = z
        if weight is None:
            self.var = np.ones_like(self.z)
        else:
            self.var = return_var_map(weight, z).flatten()

        # Spline params for saunerie.
        self.order = order
        self.nx = nx
        self.ny = ny
        self.limit = limit
        self.bs = bspline.CardinalBSpline2D(nx=self.nx, ny=self.ny,
                                            x_range=(-self.limit, self.limit),
                                            y_range=(-self.limit, self.limit), 
                                            x_order=self.order, y_order=self.order)
        # hyperparamaters
        self.theta = None
        self._J = None

    def linear_brut(self, J, y, w=None):
        if w is None:
            w = np.eye(len(y))
        else:
            w = np.eye(len(w)) * w
        X = J.todense()
        T = np.dot(X.T, w.dot(X))
        T_inv = np.linalg.inv(T)
        B = np.dot(X.T, w.dot(np.matrix(y).T))
        return np.array((np.dot(T_inv, np.matrix(B))).T).squeeze()
 
    def solve(self):
        self._J = self.bs.eval(self.x, self.y)
        self.theta = self.linear_brut(self._J, self.z, w=1./self.var)
        
    def _get_2pcf_predict(self):
        return self._J * self.theta

    def __call__(self, X, Y=None):

        if Y is None:
            ind = np.linspace(0, len(X[:,0])-1, len(X[:,0])).astype(int)
            i1, i2 = np.meshgrid(ind, ind)
            yshift = X[:,1][i2]-X[:,1][i1]
            xshift = X[:,0][i2]-X[:,0][i1]
        else:
            ind1 = np.linspace(0, len(X[:,0])-1, len(X[:,0])).astype(int)
            ind2 = np.linspace(0, len(Y[:,0])-1, len(Y[:,0])).astype(int)
            i1, i2 = np.meshgrid(ind1, ind2)
            yshift = Y[:,1][i2]-X[:,1][i1]
            xshift = Y[:,0][i2]-X[:,0][i1]

        K = np.zeros_like(xshift)

        for i in range(len(xshift[0])):
            filtre = ((abs(xshift[i]) < self.limit) & (abs(yshift[i]) < self.limit)) 
            jgp = self.bs.eval(xshift[i][filtre], yshift[i][filtre])
            K[i][filtre] = jgp * self.theta

        # Due to the spline interp, I am not sure that
        # the kernel is symetric, so to be sure,
        # I am making it symetric.
        K = (K + K.T) / 2.
        return K

class interp_dic(object):
    
    def __init__(self, dic, key='gpu'):

        self._2pcf_dist = dic['gp_output']['%s.2pcf_dist'%(key)]
        self._2pcf_weight = dic['gp_output']['%s.2pcf_weight'%(key)]
        self._2pcf_fit = dic['gp_output']['%s.2pcf_fit'%(key)]
        self._2pcf = dic['gp_output']['%s.2pcf'%(key)]
        self._2pcf_mask = dic['gp_output']['%s.2pcf_mask'%(key)]

if __name__ == "__main__":

    dic = cPickle.load(open('../../astrometry/tests/137108_z/gp_output_137108.pkl'))
    interp = interp_dic(dic, key='gpu')
    
    spline = spline2D_hyp_search(interp._2pcf_dist[:,0], interp._2pcf_dist[:,1],
                                 interp._2pcf, weight=interp._2pcf_weight,
                                 order=4, nx=11, ny=11, 
                                 limit=980.)
    spline.solve()
    znew = spline._get_2pcf_predict()

    var = return_var_map(interp._2pcf_weight, interp._2pcf)
    plot_correlation_function(interp, other_fit=znew, specific_name_kernel='Cubic spline')

    X = np.array([dic['input_data']['u'], dic['input_data']['v']]).T
    K = spline.__call__(X)
    plt.figure(figsize=(8, 8))
    plt.imshow(K, vmin=-22, vmax=22, cmap=plt.cm.seismic)

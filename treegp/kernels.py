"""
.. module:: kernel
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import _check_length_scale
from .bspline import CardinalBSpline2D

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

def eval_kernel(kernel):
    """
    Some import trickery to get all subclasses 
    of sklearn.gaussian_process.kernels.Kernel
    into the local namespace without doing 
    "from sklearn.gaussian_process.kernels import *"
    and without importing them all manually.

    Example:
    kernel = eval_kernel("RBF(1)") instead of
    kernel = sklearn.gaussian_process.kernels.RBF(1)
    """
    def recurse_subclasses(cls):
        out = []
        for c in cls.__subclasses__():
            out.append(c)
            out.extend(recurse_subclasses(c))
        return out
    clses = recurse_subclasses(Kernel)
    for cls in clses:
        module = __import__(cls.__module__, globals(), locals(), cls)
        execstr = "{0} = module.{0}".format(cls.__name__)
        exec(execstr, globals(), locals())

    from numpy import array

    try:
        k = eval(kernel)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to evaluate kernel string {0!r}.  "
                           "Original exception: {1}".format(kernel, e))

    if type(k.theta) is property:
        raise TypeError("String provided was not initialized properly")
    return k

class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a radial basis function (essentially a
    squared exponential or Gaussian) but with arbitrary anisotropic covariance.
    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length.
    For optimization, it's necessary to reparameterize the inverse covariance matrix in such a way
    as to ensure that it's always positive definite.  To this end, we define `theta` (abbreviated
    `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicRBF.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)

        if eval_gradient:
            if self.hyperparameter_cholesky_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # dK_pq/dth_k = -0.5 * K_pq *
                #               ((x_p_i-x_q_i) * dInvLam_ij/dth_k * (x_q_j - x_q_j))
                # dInvLam_ij/dth_k = dL_ij/dth_k * L_ij.T  +  L_ij * dL_ij.T/dth_k
                # dL_ij/dth_k is a matrix with all zeros except for one element.  That element is
                # L_ij if k indicates one of the theta parameters landing on the Cholesky diagonal,
                # and is 1.0 if k indicates one of the thetas in the lower triangular region.
                L_grad = np.zeros((self.ntheta, self.ndim, self.ndim), dtype=float)
                L_grad[(np.arange(self.ndim),)+self._d] = self._L[self._d]
                L_grad[(np.arange(self.ndim, self.ntheta),)+self._t] = 1.0

                half_invLam_grad = np.dot(L_grad, self._L.T)
                invLam_grad = half_invLam_grad + np.transpose(half_invLam_grad, (0, 2, 1))

                dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_grad = np.einsum("ijk,lkm,ijm->ijl", dX, invLam_grad, dX)
                K_gradient = -0.5 * K[:, :, np.newaxis] * dist_grad
                return K, K_gradient
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds


class VonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """VonKarman kernel.

    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                         special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            K = squareform(K)

            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")

            dists = cdist(X, Y, metric='euclidean')
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = ((dists[Filter]/length_scale)**(5./6.) *
                       special.kv(5./6.,2*np.pi*dists[Filter]/length_scale))
            lim0 = special.gamma(5./6.) / (2 * (np.pi**(5./6.)))
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                raise ValueError(
                    "Gradient can only be evaluated with isotropic VonKarman kernel for the moment.")
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

class AnisotropicVonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a Von-Karman correlation function
    with an arbitrary anisotropic covariance. While the parameter for this kernel,
    an inverse covariance matrix, can be specified directly with the `invLam` kwarg,
    it may be more convenient to instead specify a characteristic scale-length for each axis
    using the `scale_length` kwarg.  Note that a list or array is required so that the dimensionality
    of the kernel can be determined from its length. For optimization, it's necessary to reparameterize
    the inverse covariance matrix in such a way as to ensure that it's always positive definite.
    To this end, we define `theta` (abbreviated `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicVonKarman.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        from scipy import special
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            K = squareform(K)
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can not be evaluated.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            Filter = (dists != 0.)
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] **(5./6.) *  special.kv(5./6., 2*np.pi * dists[Filter])
            lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
            if np.sum(Filter) != len(K[0])*len(K[:,0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            raise ValueError(
                "Gradient can not be evaluated.")
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds

class Spline2D(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, pcf_x=None, pcf_y=None, pcf=None,
                 weight=None, order=4,
                 nx=11, ny=11, limit=980.):

        # 2-point correlation functions, output
        # of TreeCorr using 2D fit.
        self.pcf_x = pcf_x
        self.pcf_y = pcf_y
        self.pcf = pcf
        if weight is None:
            self.var = np.ones_like(self.pcf)
        else:
            self.var = return_var_map(weight, pcf).flatten()

        # Spline params for saunerie.
        self.order = order
        self.nx = nx
        self.ny = ny
        self.limit = limit
        self.bs = CardinalBSpline2D(nx=self.nx, ny=self.ny,
                                    x_range=(-self.limit, self.limit),
                                    y_range=(-self.limit, self.limit),
                                    x_order=self.order, y_order=self.order)
        # hyperparamaters
        self._theta = []
        self._J = None

    def init_pcf(self, pcf_x, pcf_y, pcf, weight=None):

        self.pcf_x = pcf_x
        self.pcf_y = pcf_y
        self.pcf = pcf
        if weight is not None:
            self.var = return_var_map(weight, pcf).flatten()

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
        self._J = self.bs.eval(self.pcf_x, self.pcf_y)

        ## Try to make it positive definite by going in 
        ## in Fourier space.
        from scipy import fftpack
        N = int(np.sqrt(len(self.pcf)))
        ps = fftpack.fft2(self.pcf.reshape((N, N)))
        ps = fftpack.fftshift(ps)
        ps = abs(ps)
        ps = ps.reshape(N**2)

        def _smooth(y, kernel='RBF(2.5)', limit = 9.):
            import treegp
            N = int(np.sqrt(len(y)))
            Filtre = y > np.max(y)/100. * limit
            noise_level = np.std(y[~Filtre])
            y[~Filtre] = 0
            X = np.linspace(0., len(y)-1., len(y)).reshape((len(y),1))

            gp = treegp.GPInterpolation(kernel=kernel, optimize=False,
                                        optimizer='two-pcf', anisotropic=True,
                                        normalize=False, nbins=42, min_sep=0.,
                                        max_sep=42, robust_fit=True,
                                        p0=None)
            gp.initialize(X, y, y_err=np.ones_like(y)*noise_level)
            gp.solve()
            predict = gp.predict(X, return_cov=False)
            Filtre = predict>0
            predict[~Filtre] = 0
            return predict.reshape((N,N))

        ps_predict = _smooth(ps, kernel='RBF(2.)', limit = 9.)
        pcf_clean = fftpack.ifft2(ps_predict)
        pcf_clean = fftpack.fftshift(pcf_clean)
        pcf_clean = abs(pcf_clean)
        pcf_clean = pcf_clean.reshape(N**2)
        pcf_clean = _smooth(pcf_clean, kernel='RBF(2.)', limit = 8.)
        pcf_clean = pcf_clean.reshape(N**2)

        #self._theta = self.linear_brut(self._J, self.pcf, w=1./self.var)
        self._theta = self.linear_brut(self._J, pcf_clean, w=1./self.var)

    def _get_2pcf_predict(self):
        return self._J * self._theta

    def __call__(self, X, Y=None, eval_gradient=False):

        if eval_gradient:
                raise ValueError("Gradient can not be evaluated.")

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
        self.xshift = xshift
        self.yshift = yshift
        for i in range(len(xshift[:,0])):
            # set to zero covariance when outside the box of
            # spline definition.
            filtre = ((abs(xshift[i]) < self.limit) & (abs(yshift[i]) < self.limit))
            jgp = self.bs.eval(xshift[i][filtre], yshift[i][filtre])
            K[i][filtre] = jgp * self._theta

        if Y is None:
            # Due to the spline interp, I am not sure that
            # the kernel is symetric, so to be sure,
            # I am making it symetric.
            K = (K + K.T) / 2.
        return K

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

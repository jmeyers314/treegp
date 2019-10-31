"""
Cardinal Spline, took from the saunerie package.
Develop by Nicolas Regnault originaly.
Implementation of 1-D and 2-D B-spline bases.


.. note: on the grids on which the B-Splines are defined
      -----+------+--- ... --+-----+--- ... -----+-----+--
          -k    -k+1         0     1             N    
                             [ specified by user ]
           [       internal, extended grid       ]

.. examples:
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.misc import comb


class CardinalBSpline(object):
    """
    Implements a 1D-BSpline basis of arbitrary order, 
    defined on a grid of evenly spaced knots.
    """
    def __init__(self, n=10, x_range=(0.,1.), order=4):
        """
        By default, compute cardinal BSplines of order 4 
        (i.e. CubicBSplines)
        """
        self.order = order 
        self.n_knots = n
        self.nj = self.n_knots + self.order - 1
        self.range = x_range
        self.dx = (x_range[1]-x_range[0]) / n 

    def __len__(self):
        """
        return the size of the basis
        """
        return self.nj
        
    def _start(self, xi, dtype):
        """
        compute the 1-st order splines given the xi (step functions)
        
        Args:
          xi: (ndarray)
              array of x's, rescaled so that the knots are at integer positions. 

        Returns:
          a tuple (k, xi, i, j, b, om) where 
            - k is the order of the spline (1) 
            - xi are the rescaled x's (reference to the arguments)
            - i  are the line indexes of the jacobian matrix (i = arange(len(x)))
            - j  are the column indexes of the jacobian matrix 
            - b  are the spline values 
            - om are the recursion coefficients: \omega_{j,k+1} = (x - j) / k
        """
        N = len(xi)
        i = np.arange(N)
        j = np.floor(xi)
        b = np.ones(N)
        om = (xi - j).astype(dtype)
        return [(1, xi, i, j, b, om)]    
    
    def _merge(self, (left, right)):
        """
        auxiliary function, called while blossoming up. 
        
        compute the B^k spline from a pair of adjacent B^{k-1}
        splines. The recursion formula is:
        
        .. math::
             B^k_{j}(x) = \omega^k_j(x) B^{k-1}_{j}(x) + (1-\omega^{k}_{j+1}) B^{k-1}_{j+1}(x)

        with 

        .. math::
             \omega^k_j(x) = \frac{x-t_j}{t_{j+k-1}-t_{j}}
             
        Args:
            left (tuple or None):  (k, xi, i,m j, b, om)
            right (tuple or None): (k, xi, i,m j, b, om)
              
        Returns:
            new tuple, of the form (k+1, xi, i, j, b, om) of order k+1
        """
        if right is None:
            k, xi, i, j, b, om = left
            b = b * om
            om = (xi - j) / (k+1.)
            return (k+1, xi, i, j, b, om)
        elif left is None:
            k, xi, i, j, b, om = right
            b = b * (1.-om)
            om = (xi - j + 1.) / (k+1.)
            return (k+1, xi, i, j-1, b, om)
        k_l, xi, i, j_l, b_l, om_l = left
        k_r, xi, i, j_r, b_r, om_r = right
        b = om_l*b_l + (1.-om_r)*b_r
        om = (xi-j_l) / (k_l + 1.)
        return (k_l+1, xi, i, j_l, b, om)
    
    def _blossom(self, bvlist):
        """
        blossom: compute recursively the k-th order splines from the (k-1)-th order splines.
        """
        l = [None] + bvlist + [None]
        l = zip(l[:-1], l[1:])
        return [self._merge(b) for b in l]
    
    def get_xi(self, x):
        """
        return a copy of the x' array, 
        scaled so that the knots are at integer locations 

        The spline corresponding to the first node is at location i=0.
        This means that x.min() is at location i=(order-1).
        """
        return (x-self.range[0]) / self.dx + self.order - 1.
    
    def eval(self, x, dtype=np.float32):
        """
        evaluate the value of the basis functions for each element of x
        
        Args:
          x : (ndarray of floats) 
              the x-values 
       
        Returns:
          B: (scipy.sparse.coo_matrix) 
                a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
                containing the basis values: B_{ij} = B_j(x_i)
                
        Note:
          This method is used by the 2D-version of the CardinalBSplines, which 
          assumes that the triplets (i,j,v) are sorted like they are here, i.e.: 
 
                [i=0, j=0, B_0(x0)]
                [i=1, j=0, B_0(x1)]
                      ...
                [i=0, j=1, B_1(x0)]
                      ...          
        """
        xi = (x-self.range[0]) / self.dx + self.order - 1.
        s = self._start(xi, dtype)
        for i in xrange(1,self.order):
            s = self._blossom(s)
        i = np.hstack([b[2] for b in s])
        j = np.hstack([b[3] for b in s])
        v = np.hstack([b[-2] for b in s])
        J = coo_matrix((v, (i,j)), dtype=dtype, shape=(len(x), self.nj))
        return J
    
    def deriv(self, x, dtype=np.float32):
        """
        evaluate the derivatives of the basis functions for each element of x
        
        Args:
           x: (ndarray of floats) 
       
         Returns:
            a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
            (contains the values of the derivatives)        
        """
        xi = (x-self.range[0]) / self.dx + self.order - 1.

        # lower order splines (k-1)
        s = self._start(xi, dtype)
        for i in xrange(1,self.order-1):
            s = self._blossom(s)
            
        # add 
        N = len(x) ; z = np.zeros(N)
        k0, xi, i, j0, b0, om0 = s[0]
        k1, xi, i, j1, b1, om1 = s[-1]
        s = [(k0, xi, i, j0-1, z, z)] + s + [(k1, xi, i, j1+1, z, z)]
            
        # and compute the differences B_j^k-1 - B_j+1^k-1
        l = zip(s[0:-1], s[1:])
        s = []
        for low, high in l:
            kl, xi, i, jl, bl, oml = low
            kh, xi, i, jh, bh, omh = high
            s.append((kl, xi, i, jl, bl-bh, oml))
            
        i = np.hstack([b[2] for b in s])
        j = np.hstack([b[3] for b in s])
        v = np.hstack([b[-2] for b in s])
        J = coo_matrix((v, (i,j)), dtype=dtype, shape=(N, self.nj))
        # checked with numerical derivative that the self.dx is really needed
        return J / self.dx  

    def deriv_m(self, x, m=2, dtype=np.float32):
        """
        evaluate the m-th derivatives of the basis functions for each element of x
        
         Args:
          x: (ndarray of floats) 
       
         Returns:
           a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
            (contains the values of the derivatives)        
        """
        xi = (x-self.range[0]) / self.dx + self.order - 1.

        # splines of order k-m 
        s = self._start(xi, dtype)
        for i in xrange(1,self.order-m):
            s = self._blossom(s)
            
        # add m zero elements before and after the structure 
        N = len(x) ; z = np.zeros(N)
        k0, xi, i, j0, b0, om0 = s[0]
        k1, xi, i, j1, b1, om1 = s[-1]
        s_pre = []
        s_post = []
        for q in xrange(m):
            s_pre.append([k0, xi, i, j0-m+q, z, z])
            s_post.append([k0, xi, i, j1+q+1, z, z])
        s = s_pre + s + s_post
        #        s = [(k0, xi, i, j0-1, z, z)]*m + s + [(k1, xi, i, j1+1, z, z)]*m # BUG !

        # now, we want to compute  \sum_p=0^m f[p] B_{j+p,k-m}(x)
        # with f[p] = (-1)^p C_m^p 
        # First, we compute the bvfilter f[p] itself 
        f = comb(m, np.arange(m+1))
        f[1::2] *= -1. 
        # then, we stack the basis values into one single array 
        l = []
        bvalues = np.vstack([u[-2] for u in s]).T
        for k in xrange(len(s)-m):
            kl, xi, i, jl, bl, oml = s[k]
            bv = np.sum(f * bvalues[:,k:k+m+1], axis=1)
            l.append((kl, xi, i, jl, bv, oml))
            
        i = np.hstack([b[2] for b in l])
        j = np.hstack([b[3] for b in l])
        v = np.hstack([b[-2] for b in l])
        J = coo_matrix((v, (i,j)), dtype=dtype, shape=(N, self.nj))
        return J / self.dx

    def gram(self, dtype=np.float64):
        """
        Compute the gramian matrix of the base elements:

        .. math::
            G_{ij} = \int B_i(x) B_j(x) dx
        """
        nk = self.n_knots        
        # Gaussian quadrature points and weights
        p, w = leggauss(self.order) 
                
        # points
        #        x = np.arange(self.range[0], self.range[1]+self.dx, self.dx)
        x = np.linspace(self.range[0], self.range[1], self.n_knots+1)
        ak = np.repeat(0.5*(x[1:]-x[:-1]), self.order)
        bk = np.repeat(0.5*(x[1:]+x[:-1]), self.order)
        pp = np.tile(p, nk)
        pp = ak*pp+bk
        
        # weights
        ww = np.tile(w, nk)
        N = len(ww)
        i = np.arange(N)
        W = coo_matrix((ak*ww, (i,i)), shape=(N,N))
        
        B = self.eval(pp, dtype=dtype)
        return B.T * W * B
                
    def integral(self, x):
        """
        Also have a look at this: 
        http://www.sciencedirect.com/science/article/pii/S009630030500189X
        http://imamat.oxfordjournals.org/content/17/1/37.abstract
        """
        pass

    def __call__(self, x, b, deriv=0, dtype=np.float32):
        """
        Syntaxic sugar so that if bs is a bspline
        bs(x, b) returns the evaluated spline function on the grid x
        with spline parameters b
        
        deriv: set it to the degree of the derivation you expect.
        deriv=0 is similar default and returns the spline       
        """
        if deriv == 0:
	    A = self.eval(x, dtype=dtype)
        elif deriv == 1:
            A = self.deriv(x, dtype=dtype)
        else:
            A = self.deriv_m(x, m=deriv, dtype=dtype )

        return (A * b)

    def linear_fit(self, x, y):
        """
        Makes a linear fit of the spline 
        
        x is the list of abscissa (not necessarily ordered)
        y is the corresponding list of data

        Make sure that all x values are within the bounds of the bspline you defined.        

        FIXME: add the weights
        """
        # essential to have float64, otherwise the solver complains
        J = self.eval(x, dtype=np.float64)
        JtJ = J.T.dot(J)
        return sparse.linalg.spsolve(JtJ, J.T.dot(y))
        #        return np.linalg.solve((A.T * A).todense(), A.T * y)    

class CardinalBSpline2D(object):
    """
    2D-BSpline basis of arbitrary order, 
    defined on a grid of evenly spaced knots.
    
    The basis is the cross-product of two BSpline bases, along x and y:
    
    .. math::
      B_{ij}(x,y) = B_i(x) \times B_j(y)
      
    """
    def __init__(self, 
                 nx=10, x_range=(0.,1.), x_order=4,
                 ny=10, y_range=(0.,1.), y_order=4):
        self.bx = CardinalBSpline(n=nx, x_range=x_range, order=x_order)
        self.by = CardinalBSpline(n=ny, x_range=y_range, order=y_order) # there is a bug here, no ? 
        self.nj = self.bx.nj * self.by.nj

    def __len__(self):
        """
        return the size of the basis
        """
        return self.nj

    def _cross(self, N, i, jx, vx, jy, vy, dtype=np.float32):
        """
        compute the cross-product: B_{ij}(x,y) = B_i(x) \times B_j(y)

        Args:
          N: (int) 
             number of points (N=len(x))
          j: ndarray of ints 
             a ndarray containing the row-indices in the jacobian matrix
          jx: ndarray of ints 
              the column indices in the matrix returned by the x-basis 
          vx: ndarray of floats, 
              the values B_j(x)
          jy: ndarray of ints 
              the column indices in the matrix returned by the y-basis 
          vy: ndarray of floats
              the values B_j(y)

        Returns:
          J: (scipy.sparse.coo_matrix) 
              the values of the cross-product as a (N,n) sparse jacobian matrix, 
              [N is the number of points, n the size of the 2D-basis]

        Note:
          This implementation makes an assumption on how the return value 
          of ``CardinalBSpline.eval'' is sorted internally.  
          See the documentation of this routine above.           
        """
        i  = i.reshape(-1,N).T.repeat(self.bx.order,axis=1)
        jx = jx.reshape(-1,N).T.repeat(self.bx.order,axis=1)
        vx = vx.reshape(-1,N).T.repeat(self.bx.order,axis=1)
        jy = np.tile(jy.reshape(-1,N).T, (1,self.by.order))
        vy = np.tile(vy.reshape(-1,N).T, (1,self.by.order))
        
        data = vx * vy
        i = i.ravel()
        j = jy.ravel()*self.bx.nj + jx.ravel()
        
        return coo_matrix((data.ravel(), (i,j)), 
                          shape=(N,self.nj), dtype=dtype)        

    def eval(self, x, y, dtype=np.float32):
        """
        evaluate and return the values of the basis functions for (x,y)
        
        Args:
          x: (ndarray of floats) 
              x-coordinates of the entry points
          y: (ndarray of floats) 
              y-coordinates of the entry points

        Returns:
          B: (scipy.sparse.coo_matrix)
              a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
              containing the basis values: B_{ij} = B_j(x_i)
        """
        if len(x) != len(y): 
            raise ValueError('x and y should have the same length')        
        N = len(x)
        Jx = self.bx.eval(x, dtype=dtype)
        ix, jx, vx = Jx.row, Jx.col, Jx.data
        Jx = None
        Jy = self.by.eval(y, dtype=dtype)        
        iy, jy, vy = Jy.row, Jy.col, Jy.data
        Jy = None
        #        embed()
        return self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)


    def gradient(self, x, y, dtype=np.float32):
        """
        evaluate and return the derivatives vs. x and y of the basis functions for (x,y)
        
        Args:
          x: (ndarray of floats) 
              x-coordinates of the entry points
          y: (ndarray of floats) 
              y-coordinates of the entry points

        Returns:
          dvdx: (scipy.sparse.coo_matrix)
                 a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
                 containing the values: B_{ij} = B_j'(x_i) * B_j(y_i)
          dvdy: (scipy.sparse.coo_matrix)
                 a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
                 containing the values: B_{ij} = B_j(x_i) * B_j'(y_i)
        """
        if len(x) != len(y): 
            raise ValueError('x and y should have the same length')        
        N = len(x)
        
        Jxp = self.bx.deriv(x, dtype=dtype)
        ix, jx, vx = Jxp.row, Jxp.col, Jxp.data
        Jxp = None
        Jy  = self.by.eval(y, dtype=dtype)
        iy, jy, vy = Jy.row, Jy.col, Jy.data
        Jy = None
        ddx = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        ix = jx = vx = None
        iy = jy = vy = None
        
        Jx  = self.bx.eval(x, dtype=dtype)
        ix, jx, vx = Jx.row, Jx.col, Jx.data
        Jx = None
        Jyp = self.by.deriv(y, dtype=dtype)
        iy, jy, vy = Jyp.row, Jyp.col, Jyp.data
        Jyp = None
        ddy = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        return ddx, ddy        

    def hessian(self, x, y, dtype=np.float32):
        """
        evaluate and return the partial second derivatives of the
        basis functions.

        Args:
          x: (ndarray of floats) 
              x-coordinates of the entry points
          y: (ndarray of floats) 
              y-coordinates of the entry points

        Returns:
          d2Bdx2, d2Bdy2, d2Bdxy : (scipy.sparse.coo_matrix)
              a tuple of 3 sparse, N x p jacobian matrices [N=len(x), p=len(self)]
              containing second derivatives of the basis values.
        """
        if len(x) != len(y): 
            raise ValueError('x and y should have the same length')        
        N = len(x)
        
        Jxp = self.bx.deriv_m(x, m=2, dtype=dtype)
        ix, jx, vx = Jxp.row, Jxp.col, Jxp.data
        Jxp = None
        Jy  = self.by.eval(y, dtype=dtype)
        iy, jy, vy = Jy.row, Jy.col, Jy.data
        Jy = None
        ddx2 = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        ix = jx = vx = None
        iy = jy = vy = None
        
        Jx = self.bx.eval(x, dtype=dtype)
        ix, jx, vx = Jx.row, Jx.col, Jx.data 
        Jx = None
        Jyp = self.by.deriv_m(y, m=2, dtype=dtype)
        iy, jy, vy = Jyp.row, Jyp.col, Jyp.data
        Jyp = None
        ddy2 = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)

        ix = jx = vx = None
        iy = jy = vy = None
        
        Jxp = self.bx.deriv_m(x, m=1, dtype=dtype)
        ix, jx, vx = Jxp.row, Jxp.col, Jxp.data
        Jxp = None
        Jyp = self.by.deriv_m(y, m=1, dtype=dtype)
        iy, jy, vy = Jyp.row, Jyp.col, Jyp.data
        Jyp = None
        ddxy = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        return ddx2, ddy2, ddxy
                

    def laplacian(self, x, y, dtype=np.float32):
        """
        evaluate and return the laplacian of the basis functions. 

        Args:
          x: (ndarray of floats) 
              x-coordinates of the entry points
          y: (ndarray of floats) 
              y-coordinates of the entry points

        Returns:
          L : (scipy.sparse.coo_matrix)
              a sparse, N x p jacobian matrix [N=len(x), p=len(self)]
              containing the laplacian of basis values: L_{ij} = \Delta B_j(x_i)
        """
        if len(x) != len(y): 
            raise ValueError('x and y should have the same length')        
        N = len(x)
        
        Jxp = self.bx.deriv_m(x, m=2, dtype=dtype)
        ix, jx, vx = Jxp.row, Jxp.col, Jxp.data
        Jxp = None
        Jy  = self.by.eval(y, dtype=dtype)
        iy, jy, vy = Jy.row, Jy.col, Jy.data
        Jy = None
        ddx2 = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        ix = jx = vx = None
        iy = jy = vy = None
        
        Jx = self.bx.eval(x, dtype=dtype)
        ix, jx, vx = Jx.row, Jx.col, Jx.data 
        Jx = None
        Jyp = self.by.deriv_m(y, m=2, dtype=dtype)
        iy, jy, vy = Jyp.row, Jyp.col, Jyp.data
        Jyp = None
        ddy2 = self._cross(N, ix, jx, vx, jy, vy, dtype=dtype)
        
        return ddx2 + ddy2

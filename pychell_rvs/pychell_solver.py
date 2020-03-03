# Contains the custom Nelder-Mead algorithm
import numpy as np
import sys
eps = sys.float_info.epsilon # For Amoeba xtol and tfol
import time
import pdb
from numba import njit, jit, prange
stop = pdb.set_trace

# If custom_subspace is set, n_sub_calls is ignored.
# x0 is all parameters (varied and non)
# vlb, vub are also full arrays
# vp is the array of indices for which parameters are varied
def simps(x0, foo, vlb=None, vub=None, vp=None, min_f_evals=None, max_f_evals=None, xtol=1E-4, ftol=1E-4, n_sub_calls=None, no_improv_break=3, args_to_pass=None):

    # Default input
    if vp is None:
        vp = np.arange(x0.size).astype(int)
    if vlb is None or vub is None:
        vlb = np.full(x0.size, -np.inf)
    if vub is None or vub is None:
        vub = np.full(x0.size, np.inf)
    if max_f_evals is None:
        max_f_evals = x0[vp].size * 500
    if n_sub_calls is None:
        n_sub_calls = vp.size

    # Sanity
    if np.any(vub - vlb < 0):
        ind = np.where(vub-vlb < 0)[0]
        sys.exit("ERROR: vub > vlb at i=" + str(ind))
    if np.any(x0[vp] > vub[vp]):
        ind = np.where(x0[vp] > vub[vp])[0]
        sys.exit("ERROR: x0 > vub at i=" + str(ind))
    if np.any(x0[vp] < vlb[vp]):
        ind = np.where(x0[vp] < vlb[vp])[0]
        sys.exit("ERROR: x0 < vlb at i=" + str(ind))

    # Save the initial full simplex guess
    xdefault_og = np.copy(x0) # the initial parameters (varied and not varied)

    # Constants
    step = 0.5
    n = xdefault_og.size
    np1 = n + 1
    nv = vp.size # for full simplex calls
    nvp1 = nv + 1
    penalty = 1E6

    # Initialize the simplex
    fcalls = 0
    x0v = x0[vp]
    vlbv = vlb[vp]
    vubv = vub[vp]
    right = np.zeros(shape=(nv, nvp1), dtype=float)
    left = np.tile(x0v, (nvp1, 1)).T
    diag = np.diag(0.5 * x0v)
    right[:, :-1] = diag
    simplex = left + right
    '''
    simplex = np.empty(shape=(nv, nvp1), dtype=float)
    simplex[:, 0] = np.copy(x0)
    for i in range(nv):
        xi = np.copy(x0)
        if np.isposinf(vub[vp[i]]):
            xi[i] = x0[i]*0.5
        else:
            xi[i] = (vub[vp[i]] - x0[i]) * 0.9 + x0[i]
        simplex[:, i] = xi
    fcalls += nvp1
    '''
    # Default is the initial guess
    xmin = np.copy(xdefault_og) # The current best x min from the initial parameters, updated
    fmin = np.inf # let the current fmin be infinity, updated

    dx = np.inf # breaks when less than xtol, updated

    i = 0 # keeps track of full calls to ameoba
    while i < n_sub_calls and dx >= xtol:

        # Perform Ameoba call for all parameters
        xdefault = np.copy(xmin) # the new default is the current min (varied and non varied)
        y, fmin, fcallst = ameoba(simplex, foo, xdefault, vp, vlbv, vubv, no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        xmin[vp] = y # Only update all varied parameters, y only contains varied parameters
        
        if i == n_sub_calls - 1 and vp.size > 5:
            break
        if n_sub_calls == 1:
            break
        if n_sub_calls == 2:
            simplex[:, 0] = xmin
        else:
            # Perform Ameoba call for dim=2 subspaces
            for j in range(nv-1):
                j1 = np.array([j, j+1])
                simplex_sub = np.array([[x0v[j1[0]], x0v[j1[1]]], [y[j1[0]], y[j1[1]]], [x0v[j1[0]], y[j1[1]]]]).T
                xdefault = np.copy(xmin)
                # Update subspace (subsapce )
                y[j1], fmin, fcallst = ameoba(simplex_sub, foo, xdefault, np.array([vp[j1[0]], vp[j1[1]]]), vlb[vp[j1]], vub[vp[j1]], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
                fcalls += fcallst
                simplex[:, j] = y
                xmin[vp] = y

        # Perform the last pairs of pts
        j1 = np.array([0, -1])
        simplex_sub = np.array([[x0[j1[0]], x0[j1[1]]], [y[j1[0]], y[j1[1]]], [x0[j1[0]], y[j1[1]]]]).T
        xdefault = np.copy(xmin)
        y[j1], fmin, fcallst = ameoba(simplex_sub, foo, xdefault, np.array([vp[j1[0]], vp[j1[1]]]), vlb[vp[j1]], vub[vp[j1]], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        xmin[vp] = y
        simplex[:, -2] = y

        if nv > 3:
            j1 = np.array([1, -2])

        simplex_sub = np.array([[x0[j1[0]], x0[j1[1]]], [y[j1[0]], y[j1[1]]], [y[j1[0]], x0[j1[1]]]]).T
        xdefault = np.copy(xmin)
        y[j1], fmin, fcallst = ameoba(simplex_sub, foo, xdefault, np.array([vp[j1[0]], vp[j1[1]]]), vlb[vp[j1]], vub[vp[j1]], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        xmin[vp] = y
        simplex[:, -1] = y

        i += 1
        dx = np.max(tol(np.min(simplex, axis=1), np.max(simplex, axis=1)))

    return np.array([xmin, fmin, fcalls], dtype=object)

# If custom_subspace is set, n_sub_calls is ignored.
def simps_super(x0, foo, vlb=None, vub=None, vp=None, min_f_evals=None, max_f_evals=None, xtol=1E-4, ftol=1E-4, n_sub_calls=None, custom_subspace=None, no_improv_break=3, args_to_pass=None):

    # Save the initial full simplex guess
    xdefault_og = np.copy(x0)

    # Constants
    step = 0.5
    n = xdefault_og.size
    np1 = n + 1
    nv = vp.size # for full simplex calls
    nvp1 = nv + 1
    penalty = 1E6

    # Default input
    if vp is None:
        vp = np.arange(x0.size).astype(int)
    if vlb is None or vub is None:
        vlb = np.full(x0.size, -np.inf)
    if vub is None or vub is None:
        vub = np.full(x0.size, np.inf)
    if max_f_evals is None:
        max_f_evals = x0[vp].size * 200
    if n_sub_calls is None:
        n_sub_calls = vp.size
    if custom_subspace is None:
        custom_subspace = np.empty(nv, dtype=np.ndarray)
        for i in range(nv):
            custom_subspace[i] = np.array([i, (i+1)%nv])

    # Sanity
    if np.any(vub - vlb < 0):
        ind = np.where(vub-vlb < 0)[0]
        sys.exit("ERROR: vub > vlb at i=" + str(ind))
    if np.any(x0[vp] > vub[vp]):
        ind = np.where(x0[vp] > vub[vp])[0]
        sys.exit("ERROR: x0 > vub at i=" + str(ind))
    if np.any(x0[vp] < vlb[vp]):
        ind = np.where(x0[vp] < vlb[vp])[0]
        sys.exit("ERROR: x0 < vlb at i=" + str(ind))

    # Init
    fcalls = 0
    x0 = x0[vp]
    right = np.zeros(shape=(nv, nvp1), dtype=float)
    left = np.tile(x0, (nvp1,1)).T
    diag = np.diag(0.5*x0)
    right[:, :-1] = diag
    simplex = left + right
    '''
    simplex = np.empty(shape=(nv, nvp1), dtype=float)
    simplex[:, 0] = np.copy(x0)
    for i in range(nv):
        xi = np.copy(x0)
        if np.isposinf(vub[vp[i]]):
            xi[i] = x0[i]*0.5
        else:
            xi[i] = (vub[vp[i]] - x0[i]) * 0.9 + x0[i]
        simplex[:, i] = xi
    fcalls += nvp1
    '''

    # Default is the initial guess
    xmin = np.copy(xdefault_og)
    fmin = np.inf

    dx = np.inf

    for i in range(nv):
        if dx <= xtol:
            break

        # Perform Ameoba call for all parameters
        xdefault = np.copy(xmin)
        y, fmin, fcallst = ameoba(simplex, foo, xdefault, vp, vlb[vp], vub[vp], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
        fcalls += fcallst
        xmin[vp] = y

        for j in range(len(custom_subspace)):
            j1 = custom_subspace[j]
            nj1 = j1.size
            right = np.zeros(shape=(nj1, nj1+1), dtype=float)
            left = np.tile(y[j1], (nj1+1, 1)).T
            diag = np.diag(0.5*x0[j1])
            right[:, :-1] = diag
            simplex_sub = left + right
            xdefault = np.copy(xmin)
            y[j1], fmin, fcallst = ameoba(simplex_sub, foo, xdefault, vp[j1], vlb[vp[j1]], vub[vp[j1]], no_improv_break, max_f_evals, ftol, penalty, args_to_pass=args_to_pass)
            fcalls += fcallst
            simplex[:, j] = y
            xmin[vp] = y

        dx = np.max(tol(np.min(simplex, axis=1), np.max(simplex, axis=1)))

    return np.array([xmin, fmin, fcalls], dtype=object)

# Ameoba assumes that simplex has been modified for any unvaried parameters.
# simplex: The simplex of varied parameters, determined by subspace
# foo: the target function
# xdefault: the default values to use if not varying a parameter
# subspace: the array of par indices being varied
# vlb, vub: the lower and upper bounds, for current varied parameters. only varied parameters are compared against vlb/vub
# no_improv_break: how many times in a row convergence occurs before exiting
def ameoba(simplex, foo, xdefault, subspace, vlb, vub, no_improv_break, max_f_evals, ftol, penalty, args_to_pass=None):

    # Constants
    n = np.min(simplex.shape)
    np1 = n + 1
    alpha = 1
    gamma = 2
    sigma = 0.5
    delta = 0.5

    fvals = np.empty(np1, dtype=float)
    for i in range(np1):
        xi = simplex[:, i]
        fi = foo_wrapper(xi, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
        fvals[i] = fi
    fcalls = np1

    ind = np.argsort(fvals)
    simplex = simplex[:, ind]
    fvals = fvals[ind]
    fmin = fvals[0]
    xmin = simplex[:, 0]
    n_small_steps = 0

    x = np.empty(n, dtype=float)
    g = np.empty(n, dtype=float)

    while True:

        # Sort the vertices according from best to worst
        # Define the worst and best vertex, and f(best vertex)
        xnp1 = simplex[:, -1]
        fnp1 = fvals[-1]
        x1 = simplex[:, 0]
        f1 = fvals[0]
        xn = simplex[:, -2]
        fn = fvals[-2]
        
        # Gradient Estimate
        #xs = np.diagonal(simplex[:, :-1])
        #fs = foo_wrapper(xs)
        #g[:] = estimate_gradient(n, simplex, fvals, xs, fs)
            
        
        shrink = 0

        # break after max_iter
        if fcalls >= max_f_evals:
            break
        if tol(fmin, fnp1) > ftol:
            n_small_steps = 0
        else:
            n_small_steps += 1
        if n_small_steps >= no_improv_break:
            break


        xbar = np.average(simplex[:, :-1], axis=1)
        xr = xbar + 1 * (xbar - xnp1)
        x[:] = xr
        fr = foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
        fcalls += 1

        if fr < f1:
            xe = xbar + 2 * (xbar - xnp1)
            x[:] = xe
            fe = foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
            fcalls += 1
            if fe < fr:
                simplex[:, -1] = xe
                fvals[-1] = fe
            else:
                simplex[:, -1] = xr
                fvals[-1] = fr
        elif fr < fn:
            simplex[:, -1] = xr
            fvals[-1] = fr
        else:
            if fr < fnp1:
                xc = xbar + 0.5 * (xbar - xnp1)
                x[:] = xc
                fc = foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
                fcalls += 1
                if fc <= fr:
                    simplex[:, -1] = xc
                    fvals[-1] = fc
                else:
                    shrink = 1
            else:
                xcc = xbar + 0.5 * (xnp1 - xbar)
                x[:] = xcc
                fcc = foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
                fcalls += 1
                if fcc < fvals[-1]:
                    simplex[:, -1] = xcc
                    fvals[-1] = fcc
                else:
                    shrink = 1
        if shrink > 0:
            for j in range(1, np1):
                x[:] = x1 + 0.5 * (simplex[:, j] - x1)
                simplex[:, j] = x
                f = foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass=args_to_pass)
                fvals[j] = f
            fcalls += n

        ind = np.argsort(fvals)
        fvals = fvals[ind]
        simplex = simplex[:, ind]
        fmin = fvals[0]
        xmin = simplex[:, 0]

    # Returns only the best varied parameters
    return xmin, fmin, fcalls


def tol(a, b):
    c = (np.abs(b) + np.abs(a)) / 2
    c = np.atleast_1d(c)
    ind = np.where(c < eps)[0]
    if ind.size > 0:
        c[ind] = 1
    r = np.abs(b - a) / c
    return r

@njit
def estimate_gradient(n, simplex, fvals, xs, fs):
    g = np.empty(n, dtype=float)
    for i in range(n):
        if i%2 == 0:
            g[i] = (fvals[i-1] - fs) / (simplex[i-1, i])
        else:
            g[i] = (fvals[i+1] - fs) / (simplex[i+1, i])
    return g

def foo_wrapper(x, foo, xdefault, vlb, vub, subspace, penalty, args_to_pass):

    # Update the current varied subspace from the current simplex
    xdefault[subspace] = x
    
    # Determine which parameters are being varied in the current simplex.
    v = np.zeros(xdefault.size, dtype=bool)
    v[subspace] = 1
    
    # Call the target function
    f, c = foo(xdefault, v, *args_to_pass) # target function must be given full array of pars
    
    # Penalize the target function if pars are out of bounds or constraint is less than zero
    f += penalty*np.where((x <= vlb) | (x >= vub))[0].size
    f += penalty*(c < 0)
    return f
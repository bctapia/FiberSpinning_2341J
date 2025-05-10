'''
Copyright (c) 2025, Brandon C. Tapia
MIT License
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root


def alpha_solve(parameters, preset=None):
    '''
    Inlet Ratio
    As defined in 2. Model Formulation
    '''

    if parameters is None:
        parameters = {}

    R_inner = parameters.get('R_inner')
    R_outer = parameters.get('R_outer')

    if preset:
        result = None
    else:
        result = R_inner/R_outer

    return result

def eps_solve(parameters):
    '''
    
    '''
    R_outer = parameters.get('R_outer')
    L = parameters.get('L')
    alpha = alpha_solve(parameters)

    eps = R_outer*np.sqrt(1-alpha**2)/L

    return eps

def Re_solve(parameters, preset=None):
    '''
    Reynolds Number
    Eq. 2.24a
    '''

    if parameters is None:
        parameters = {}

    U_in = parameters.get('U_in')
    L = parameters.get('L')
    rho = parameters.get('rho')
    K = parameters.get('K')
    n = parameters.get('n')

    if preset:
        result = preset
    else:
        result = (U_in/L)**(1-n)*rho*U_in*L/K
    return result

def Ca_solve(parameters, preset=None):
    '''
    Capillary Number
    Eq. 2.24b
    '''

    if parameters is None:
        parameters = {}

    U_in = parameters.get('U_in')
    L = parameters.get('L')
    K = parameters.get('K')
    n = parameters.get('n')
    gamma = parameters.get('gamma')

    eps = eps_solve(parameters)

    if preset:
        result = preset
    else:
        result = (U_in/L)**(1-n)*K*U_in*eps/gamma
        #print('yes',result)
    return result

def D_solve(parameters, preset=None):
    '''
    Draw Ratio
    Eq. 2.24c
    '''

    if parameters is None:
        parameters = {}

    U_in = parameters.get('U_in')
    U_out = parameters.get('U_out')

    if preset:
        result = None
    else:
        result = U_out/U_in
    return result

def system(z, y, parameters, loss_function, show_scale):
    '''
    System of ODEs to solve
    '''

    u, du_dz, h, H = y # parameters to solve for

    # Gathering parameters
    Re = Re_solve(parameters)
    Ca = Ca_solve(parameters)
    n = parameters.get('n')

    # Solving for h
    right_h_H_term = - 3**((1-n)/2)*h*H/(Ca*(H-h))*du_dz**(1-n)

    dh_dz = (right_h_H_term-h**2*du_dz)/(2*h*u)

    Lz = loss_function

    if show_scale:
        print('du/dz=',dh_dz)
        print('rhight_H_h_TERM=',right_h_H_term)

    dH_dz = (right_h_H_term-H**2*du_dz-Lz)/(2*H*u)

    # Solving for u
    Re_term = Re*du_dz
    Ca_term = 1/Ca*(dh_dz+dH_dz)
    d_h_H_term = 2*H*dH_dz-2*h*dh_dz

    term1 = (Re_term-Ca_term)*(3**((n+1)/2))**-1-d_h_H_term*du_dz**n
    term2 = ((H**2-h**2)*n*du_dz**(n-1))**-1

    d2u_dz2 = term1*term2

    return [du_dz, d2u_dz2, dh_dz, dH_dz]

def shoot_steady(parameters, z_eval=None, loss_function=0.0, show_scale=False):

    # dimensionless params
    alpha = alpha_solve(parameters)
    D = D_solve(parameters)
    n = parameters.get('n')

    # initial h,H
    h0 = alpha/np.sqrt(1-alpha**2)
    H0 = 1/np.sqrt(1-alpha**2)

    if abs(n-1.0) > 1e-8:
        guess = (n/(n-1.0))*(D**((n-1.0)/n)-1.0)
    else:
        guess = D - 1.0

    def residual_vec(slope_arr):
        slope0 = float(slope_arr[0])
        if slope0 <= 0:
            return np.array([1e6])

        y0 = [1.0, slope0, h0, H0]
        sol = solve_ivp(fun=system, t_span=(0,1), y0=y0,args=(parameters, loss_function, show_scale),rtol=1e-8, atol=1e-10, max_step=0.01)

        if not sol.success:
            return np.array([1e6])

        return np.array([sol.y[0,-1] - D])


    sol_root = root(fun=residual_vec,x0=[guess],method='hybr',tol=1e-8)
    if not sol_root.success:
        raise RuntimeError('Failed with message:', sol.message)

    best_slope = float(sol_root.x[0])

    print(f'Found du = {best_slope}')

    y0 = [1.0, best_slope, h0, H0]
    sol = solve_ivp(fun=system, t_span=(0,1), y0=y0,args=(parameters, loss_function, show_scale),rtol=1e-8, atol=1e-10, dense_output=True)

    if not sol.success:
        raise RuntimeError('Failed with message:', sol.message)

    if z_eval is None:
        z_eval = np.linspace(0,1,300)

    u, du, h, H = sol.sol(z_eval)
    return z_eval, u, h, H

def res_time(z,u, parameters):
    '''
    Computes the residence time for a fluid control volume
    '''
    # this is coming from v = dz/dt -> t = integral 1/v dz
    # this means I know the time the fluid has spent at each z location
    #t_res = cumulative_trapezoid(1/u, z, initial=0.0)
    #v = np.ones_like(u)*0.007
    U_in = parameters.get('U_in')
    t_res = cumulative_trapezoid(1/(u*U_in), z, initial=0.0)
    return t_res # <- If there was no speed up this should be linear w/ z


def find_tmax(params, z, t_res):

    length = params.get('L')
    airgap = params.get('airgap')
    z1 = airgap/length
    #print(z1)
    #print(z)
    #print(np.argmin(np.abs(z-z1)))
    t_max = t_res[np.argmin(np.abs(z-z1))]
    return t_max

'''
Copyright (c) 2025, Brandon C. Tapia
MIT License
'''

import numpy as np
from scipy.integrate import solve_ivp

def annular_diffusion(params, c0, r_inner, r_outer, t_max):
    '''
    *
    '''

    c_inf = params.get('c_inf')
    D_s = params.get('D_s')

    r = np.linspace(r_inner, r_outer, len(c0))
    dr = r[1]-r[0]

    t_eval = np.linspace(0, t_max, len(c0))

    def solver(t, C):
        dCdt = np.zeros_like(C)

        dCdt[0] = 0
        dCdt[-1] = 0

        for i in range(1, len(c0)-1):
            d2C_dr2 = (C[i+1]-2*C[i]+C[i-1])/dr**2
            dC_dr = (C[i+1]-C[i-1])/(2*dr)
            dCdt[i] = D_s*(d2C_dr2+dC_dr/r[i])
        return dCdt

    C0 = c0

    C0  = np.copy(c0)
    C0[-1] = c_inf

    sol = solve_ivp(solver, (0, t_max), C0, t_eval=t_eval)
    C = sol.y

    C[0, :]  = c0[0]
    C[-1, :] = c_inf
    #print(sol.t)
    return r, sol.t, C

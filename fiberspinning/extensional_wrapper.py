'''
Copyright (c) 2025, Brandon C. Tapia
MIT License
'''

import flow
import diffusion
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.transforms import ScaledTranslation

parameters = {
    'R_inner':1.3208/1000,#0.4
    'R_outer':1.778/1000, # 1.2
    'U_in': 0.00748909,
    'U_out':0.00748909*1.5,
    'L':0.3,
    'airgap': 0.01,
    'rho':1000.0,
    'K':44.6,
    'n':0.923,
    'gamma':0.0005,
    'D_s': 1E-9,
    'c_inf': 0,
    'c_0': 1-0.29
}

#print(flow.Ca_solve(parameters))

def flow_conc():
    '''
    returns concentration data for each time step
    
    '''
    z, u, h, H = flow.shoot_steady(parameters, loss_function=0.0)
    t_res = flow.res_time(z,u, parameters)
    t_max = flow.find_tmax(parameters, z, t_res)

    c_array = []
    r_array = []
    dt_array = []

    c0 = np.ones(40000)*parameters.get('c_0')

    for i, t_val in enumerate(t_res):

        if t_val < t_max and i != len(t_res) - 1 :

            dt = t_res[i+1] - t_val
            dt_array.append(t_val)
            h_use = h[i]
            H_use = H[i]

            r,t,c = diffusion.annular_diffusion(parameters, c0, h_use, H_use, dt)

            c0 = c[:,-1]
            r_array.append(r)
            c_array.append(c0)

    return dt_array, r_array, c_array

def scaling(parameters):

    flow.shoot_steady(parameters, show_scale=True)

#scaling(parameters)

def flux(parameters):
    '''
    *
    '''
    dt_array, r_array, c_array = flow_conc()


    D_s = parameters.get('D_s')
    z, u0, h, H = flow.shoot_steady(parameters)
    t_res = flow.res_time(z,u0, parameters)
    t_max = flow.find_tmax(parameters, z, t_res)
    rho = parameters.get('rho')
    epsilon = flow.eps_solve(parameters)
    length = parameters.get('L')
    airgap = parameters.get('L')
    v_in = parameters.get('U_in')

    sink_array = np.zeros(len(dt_array))

    for i, dt_val in enumerate(dt_array):

        H_use = H[i]
        C_use = c_array[i]
        ri = r_array[i]
        dr = ri[1]-ri[0]

        dCdr = (3*C_use[-1]-4*C_use[-2]+C_use[-3])/(2*dr)

        J = -D_s*dCdr
        #print(J)
        m = 2*np.pi*H[i]*J
        ##print(m)
        #print(epsilon)
        sink = m/(epsilon**2*length*v_in*rho*np.pi)

        #sink_array[i] = 2*np.pi*H[i]*J
        print('sink',sink*0.02)
    return #interp1d(z, sink_array)

#flux(parameters)

def conc(parameters):
    '''
    *
    '''
    dt_array, r_array, c_array = flow_conc()

    fig, ax = plt.subplots(1,2,figsize=(8.4,4), constrained_layout=True)
    cmap = mpl.colormaps.get_cmap('viridis').resampled(len(dt_array)+5)

    for i, val in enumerate(dt_array):
        if i % 2 == 0:
            #H_use = H[i]
            C_use = c_array[i]
            ri = r_array[i]
            ax[0].plot(ri,C_use, label=rf'$t={val:.2f} \; s$', color=cmap(i))
            ax[0].legend(frameon=False)
        C_use = c_array[i]
        ri = r_array[i]
        coloring = cmap(i)

    ax[0].plot(r_array[-1], c_array[-1], label=rf'$t={dt_array[-1]:.2f} \; s$', color=coloring)
    ax[0].legend(frameon=False)
    ax[0].set_xlabel(r'$r$ (mm)')
    ax[0].set_ylabel(r'$\phi$')
    #ax[0].set_xlim(1.4,1.5)

    ax[1].plot(r_array[-1], c_array[-1], label=rf'$t={dt_array[-1]:.2f} \; s$', color=coloring)
    ax[1].axvline(ri[-1], color='black', linestyle='--')
    ax[1].set_xlim(1.467,1.468)
    ax[1].legend(frameon=False, loc='lower left')
    ax[1].set_xlabel(r'$r$ (mm)')
    ax[1].set_ylabel(r'$\phi$')

    labels = ['a)', 'b)']
    axes   = ax.flatten()
    ax[1].ticklabel_format(style='plain', useOffset=False)

    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    for label, axi in zip(labels, axes):
        offset = ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)
        axi.text(0.0, 1.0, label,
                transform=axi.transAxes + offset,
                fontsize='medium', va='bottom')


    fig.savefig('spinners_changingh.png',
        dpi=600,
        bbox_inches='tight',
        format='png')
    #for i, dt_val in enumerate(dt_array):

    plt.show()

#conc(parameters)

def base(parameters):
    '''
    *
    '''
    #flux(parameters, r_array, c_array, dt_array)
    fig, ax = plt.subplots(1,2,figsize=(8.4,4), constrained_layout=True)
    cmap = mpl.colormaps.get_cmap('viridis').resampled(4)
    loss = 0.0
    length = parameters.get('L')
    z, u, h, H = flow.shoot_steady(parameters, loss_function=loss)
    ax[0].plot(z, u, label=f'loss={loss}', color=cmap(0))
    ax[0].set_xlabel(r'$\tilde{z}$')
    ax[0].set_ylabel(r'$\tilde{v}_x$')
    #ax[0].legend()

    r2 = flow.eps_solve(parameters)*H*length

    r1 = flow.eps_solve(parameters)*h*length
    ax[1].plot(z, r2*1000, label=r'$r_2$', color=cmap(1))
    ax[1].plot(z, r1*1000, label=r'$r_1$', color=cmap(2))
    ax[1].set_xlabel(r'$\tilde{z}$')
    ax[1].set_ylabel(r'$r_2 \; \mathrm{or} \; r_1$ (mm)')

    ax[1].legend(frameon=False)
    labels = ['a)', 'b)']
    axes   = ax.flatten()

    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    for label, axi in zip(labels, axes):
        offset = ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)
        axi.text(0.0, 1.0, label,
                transform=axi.transAxes + offset,
                fontsize='medium', va='bottom')

    #ax[1].set_xlabel(r'$\tilde{z}$')
    #ax[1].set_ylabel(r'$\tilde{h}$')
        #ax[2].legend()
    fig.savefig('spinners_changingh.png',
        dpi=600,
        bbox_inches='tight',
        format='png')
    #plt.tight_layout()
    plt.show()

def loss_change():
    '''
    *
    '''
    #flux(parameters, r_array, c_array, dt_array)
    fig, ax = plt.subplots(2,2,figsize=(8.4,8), constrained_layout=True)
    loss_array = [0.0,1E-5,1E-4,1E-3,1E-2,1E-1,0.5,0.8]
    cmap = mpl.colormaps.get_cmap('viridis').resampled(len(loss_array)+1)
    for i, loss in enumerate(loss_array):
        z, u, h, H = flow.shoot_steady(parameters, loss_function=loss)
        ax[0,0].plot(z, u, label=f'loss={loss}', color=cmap(i))
        ax[0,0].set_xlabel(r'$\tilde{z}$')
        ax[0,0].set_ylabel(r'$\tilde{v}_x$')
        #ax[0].legend()

        ax[0,1].plot(z, H, label=f'loss={loss}', color=cmap(i))
        ax[0,1].set_xlabel(r'$\tilde{z}$')
        ax[0,1].set_ylabel(r'$\tilde{H}$')
        #ax[1].legend()

        ax[1,0].plot(z, h, label=rf'$\tilde{{S}}={loss:.0e}$', color=cmap(i))
        ax[1,0].set_xlabel(r'$\tilde{z}$')
        ax[1,0].set_ylabel(r'$\tilde{h}$')
        ax[1,0].legend(prop={'size': 20})

        #ax[1,1].set_visible(False)

    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    labels = ['a)', 'b)', 'c)', 'd)']
    axes   = ax.flatten()

    for label, axi in zip(labels, axes):
        offset = ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)
        axi.text(0.0, 1.0, label,
                transform=axi.transAxes + offset,
                fontsize='medium', va='bottom')

    fig.savefig('spinners_changingh.png',
        dpi=600,
        bbox_inches='tight',
        format='png')
    #plt.tight_layout()
    plt.show()

def ten_change():
    '''
    *
    '''
    
    fig, ax = plt.subplots(2,2,figsize=(8.4,8), constrained_layout=True)
    ten_array = [0,0.00001,0.0001,0.001]
    cmap = mpl.colormaps.get_cmap('viridis').resampled(len(ten_array)+1)
    for i, ten in enumerate(ten_array):
        loss = 0.0
        parameters['gamma'] = ten
        z, u, h, H = flow.shoot_steady(parameters, loss_function=loss)
        ax[0,0].plot(z, u, label=f'S={loss}', color=cmap(i))
        ax[0,0].set_xlabel(r'$\tilde{z}$')
        ax[0,0].set_ylabel(r'$\tilde{v}_x$')
        #ax[0].legend()

        ax[0,1].plot(z, H, label=f'S={loss}', color=cmap(i))
        ax[0,1].set_xlabel(r'$\tilde{z}$')
        ax[0,1].set_ylabel(r'$\tilde{H}$')
        #ax[1].legend()

        ax[1,0].plot(z, h, label=rf'$\gamma={ten:.0e}\; \mathrm{{N\;m^2}}$', color=cmap(i))
        ax[1,0].set_xlabel(r'$\tilde{z}$')
        ax[1,0].set_ylabel(r'$\tilde{h}$')
        ax[1,0].legend(prop={'size': 20})

        #ax[1,0].set_visible(False)
    labels = ['a)', 'b)', 'c)', 'd)']
    axes   = ax.flatten()

    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    for label, axi in zip(labels, axes):
        offset = ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)
        axi.text(0.0, 1.0, label,
                transform=axi.transAxes + offset,
                fontsize='medium', va='bottom')

    fig.savefig('spinners_changingh.png',
        dpi=600,
        bbox_inches='tight',
        format='png')
    #plt.tight_layout()
    plt.show()

def n_change():
    '''
    *
    '''
    
    fig, ax = plt.subplots(2,2,figsize=(8.4,8), constrained_layout=True)
    n_array = [0.9, 0.923, 1, 1.1]
    cmap = mpl.colormaps.get_cmap('viridis').resampled(len(n_array)+1)
    for i, n_val in enumerate(n_array):
        loss = 0.0
        parameters['n'] = n_val
        z, u, h, H = flow.shoot_steady(parameters, loss_function=loss)
        ax[0,0].plot(z, u, label=f'loss={loss}', color=cmap(i))
        ax[0,0].set_xlabel(r'$\tilde{z}$')
        ax[0,0].set_ylabel(r'$\tilde{v}_x$')
        #ax[0].legend()

        
        ax[0,1].plot(z, H, label=f'loss={loss}', color=cmap(i))
        ax[0,1].set_xlabel(r'$\tilde{z}$')
        ax[0,1].set_ylabel(r'$\tilde{H}$')
        #ax[1].legend()

        ax[1,0].plot(z, h, label=rf'$n={n_val}$', color=cmap(i))
        ax[1,0].set_xlabel(r'$\tilde{z}$')
        ax[1,0].set_ylabel(r'$\tilde{h}$')
        #ax[1,0].legend(prop={'size': 20})

        ax[1,1].set_visible(False)
    
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    labels = ['a)', 'b)', 'c)', 'd)']
    axes   = ax.flatten()
    #ax[1].ticklabel_format(style='plain', useOffset=False)  #

    for label, axi in zip(labels, axes):
        offset = ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)
        axi.text(0.0, 1.0, label,
                transform=axi.transAxes + offset,
                fontsize='medium', va='bottom')
    
    fig.savefig('spinners_changingh.png',
        dpi=600,
        bbox_inches='tight',
        format='png')
    #plt.tight_layout()
    plt.show()

#conc(parameters)
#flux(parameters)
base(parameters)
#loss_change()
#ten_change()
#n_change()

import numpy
import fit_binned_nof_photons_v3 as fbp
from common_calcs import *
Lpix = 3
from numba import njit
from tqdm import trange
import os
from scipy.io import savemat


global alpha_poly, gamma_poly, r_max_poly, r_min_poly, za, ze, lr_cut, r_cube

#@njit
def polyjit(ps, x):
    val = 0
    for n,p in enumerate(ps[::-1]):
        val  = val + p*(x**n)
    return val


#@njit
def distdist_model(m, beta1, beta2 ,lr_max, bst, za, ze, lr):

    red_factor = 1e-6
    c = numpy.zeros(lr.shape)
    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    c = red_factor*m*(lr)*( 1 / (1 + numpy.exp(  ((lr - lr_max)/(0.5*beta1))   )) )**2
    c = c + red_factor*bst*numpy.exp( -(lr-(lr_max-beta1))**2/(2*beta2**2))

    c[lr > lr_cut] = numpy.nan
    c[c < 0] = 0

    return c

#@njit
def get_fraction_of_photons(r, lr_min, gamma, alpha ,lr_max, lr_cut):


    if r < 3:
        r = 3
    lr = numpy.log10(r)
    if lr < lr_min:
        nof_photons = 0
        nof_photons_per_volume = 0
    else:
        nof_photons = (1/r)*10**(distdist_model(lr_min, gamma, alpha ,lr_max, lr_cut, lr))
        nof_photons_per_volume = nof_photons/r**2


    return nof_photons_per_volume



#@njit
def get_real_space_window(real_space_window, N, rshell, lr_cut):

    r_cut = 10**(lr_cut)
    pjit = numpy.zeros(5)
    pjit[-len(r_min_poly):] = r_min_poly
    r_min = polyjit(pjit, rshell)
    if r_min < 0.001:
        r_min = 0.001
    lr_min = numpy.log10(r_min)

    pjit = numpy.zeros(5)
    pjit[-len(r_max_poly):] = r_max_poly
    r_max = polyjit(pjit, rshell)
    if r_max > r_cut:
        r_max = r_cut
    if r_max < r_min:
        r_max = r_cut
    lr_max = numpy.log10(r_max)

    pjit = numpy.zeros(5)
    pjit[-len(gamma_poly):] = gamma_poly
    gamma = polyjit(pjit, rshell)

    pjit = numpy.zeros(5)
    pjit[-len(alpha_poly):] = alpha_poly
    alpha = polyjit(pjit, rshell)
    print('r_min = {:.1f}, gamma = {:.1f}, alpha = {:.1f}, r_max = {:.1f}, r_cut = {:.1f}'.format(r_min, gamma, alpha ,r_max, r_cut))


    mid = int(N/2) + 1
    for xi in range(mid):
        for yi in range(mid):
            for zi in range(mid):
                if (xi >= yi) and (yi >= zi):
                    res = get_fraction_of_photons(r_cube[xi,yi,zi], lr_min, gamma, alpha ,lr_max, lr_cut)

                    xit =  - xi
                    yit =  - yi
                    zit =  - zi

                    for xif in [xi, xit]:
                        for yif in [yi, yit]:
                            for zif in [zi, zit]:


                                real_space_window[xif,yif,zif] = res
                                real_space_window[xif,zif,yif] = res
                                real_space_window[yif,xif,zif] = res
                                real_space_window[yif,zif,xif] = res
                                real_space_window[zif,yif,xif] = res
                                real_space_window[zif,xif,yif] = res


    return

def get_log_log_data(dists):

    log_dists = numpy.log10(dists)
    counts, bins = numpy.histogram(log_dists, bins = 50)

    bins = (bins[1:] + bins[:-1])/2
    log_counts = numpy.zeros(counts.shape)
    log_counts[counts > 0] =  numpy.log10(counts[counts > 0])
    d_counts = numpy.sqrt(counts)
    d_log_counts = numpy.zeros(d_counts.shape)
    cond = log_counts < 1e-3
    d_log_counts[~cond] = d_counts[~cond]/(counts[~cond])
    d_log_counts[cond] = 1

    return bins, log_counts, d_log_counts

def plot_res(lr_min, m, alpha, lr_max, bins, log_counts, d_log_counts, za, ze):

    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    lr_grid = numpy.linspace(numpy.min(bins), numpy.max(bins), 1000)
    ddm = distdist_model(lr_min, m, alpha, lr_max, za, ze, lr_grid)
    plt.figure(figsize = (15,7))
    fs = 25
    lw = 7
    plt.vlines(x=lr_cut, ymin=-5, ymax=10, color='silver' ,lw = lw, label='Straight line distance' )
    plt.plot(lr_grid, ddm, lw = lw, label = 'Fit, $\gamma$ = {:.2f}'.format(m), color = 'orangered')
    plt.errorbar(bins, log_counts, xerr = None, yerr =d_log_counts,  color = 'k', fmt='.', ms = 20, label = 'Binned MC resutlts',capsize = 5, capthick=3)



    plt.ylim([-0.2, numpy.max(log_counts) + 0.2])
    plt.xlabel(r'log$_{10}$(R [Mpc])', fontsize = fs)
    plt.ylabel(r'log$_{10}$(number of photons)', fontsize = fs)
    plt.legend(fontsize = fs)
    plt.show()


if __name__ == '__main__':

    p_window = '/Volumes/Backup/Cosmo/IC_and_backgrouds_py/lya_window_functions_from_model/'



    N = 128
    mid_cube = float(N)/2
    Lpix = 3
    real_space_window = numpy.zeros([N,N,N])

    x = numpy.arange(N) - mid_cube
    y = numpy.arange(N) - mid_cube
    z = numpy.arange(N) - mid_cube

    r_cube = numpy.sqrt(x[:,None,None]**2 + y[None,:,None]**2 + z[None,None,:]**2)*Lpix

    z_simul = numpy.arange(5,10)

    for z_center in z_simul:
        alpha_poly = numpy.load('fitting_results_v3/alphas/alpha_poly_za_{}.npy'.format(z_center))
        gamma_poly = numpy.load('fitting_results_v3/gammas/gamma_poly_za_{}.npy'.format(z_center))
        r_max_poly = numpy.load('fitting_results_v3/rmaxs/rmax_poly_za_{}.npy'.format(z_center))
        r_min_poly = numpy.load('fitting_results_v3/rmins/rmin_poly_za_{}.npy'.format(z_center))
        print(z_center)
        za = z_center
        for ri in range(nof_shells):
            r_shell = Lpix*(Rmax[ri] + Rmin[ri])/2


            fname = '{}za_{}_R_{}.mat'.format(p_window, z_center, ri)
            if os.path.isfile('XXX' + fname):
                pass
            else:
                z_shell = R_to_zshell(r_shell, z_center)
                ze = z_shell
                lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

                real_space_window = numpy.zeros([N,N,N])
                get_real_space_window(real_space_window, N, r_shell, lr_cut)
                real_space_window[numpy.isnan(real_space_window)] = 0
                norm = numpy.sum(real_space_window)
                print('{:.2f}, {:.2f}'.format(z_shell, norm))
                if norm > 0.000001:
                    real_space_window = real_space_window/norm
                else:
                    real_space_window = numpy.zeros([N,N,N])
                '''
                DFT
                '''
                shifted_real_space_window = numpy.fft.ifftshift(real_space_window)
                k_space_window = numpy.real(numpy.fft.fftn(shifted_real_space_window))

                '''
                Save k-space window function
                '''
                m_dict = {'windowk':k_space_window}
                savemat(fname, m_dict, do_compression=True)

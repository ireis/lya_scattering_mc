from common_calcs import *
import pandas
from tqdm import trange
import os
import matplotlib.pyplot as plt
from matplotlib_params import set_plt_params
plt = set_plt_params(plt)

from scipy import optimize
def initial_guess(x, y, dy, za, ze):

    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    #print('lr_cut = {:.2f}'.format(lr_cut))
    s = numpy.argmax(dy < 0.1*y)
    e = numpy.max([numpy.argmax(y) - 10, s+5])
    #print(s, e)
    try:
        d = (y[s+1:e+1] - y[s:e])/(x[s+1:e+1] - x[s:e])
        m = numpy.mean(d)
    except:
        print(s, e)
        m=3

    last_int = numpy.argmax(x > lr_cut)-1
    last_val = y[last_int]
    max_val = numpy.max(y)
    lr_max = x[numpy.argmax(y)]

    after_max = y[numpy.argmax(y):]
    lr_cut_data_loc = numpy.argmax(after_max == 0) + numpy.argmax(y)
    lr_cut_data = x[lr_cut_data_loc]
    #print(lr_cut_data, lr_cut)


    last_to_max = last_val/max_val
    #print(last_to_max)
    if last_to_max < 0.1:
        last_to_max = 0.1

    mid_ind = int((e+s)/2)
    mid_val = y[mid_ind]
    mid_x = x[mid_ind]

    lr_min = mid_x - mid_val/m

    pred_max = (lr_max - lr_min)*m

    alpha = (lr_cut_data - lr_max)/3
    beta2 = (lr_cut_data - lr_max)

    pred_max = m*(lr_max)*( 1 / (1 + numpy.exp(  ((lr_max - lr_cut)/(0.5*alpha))   )) ) - m*lr_min

    bst = (max_val - pred_max)
    #if bst < 0:
    #    bst = 0

    lr_a = lr_max

    lr_max_ig = lr_cut_data


    return lr_min, m, alpha, beta2, lr_max_ig, bst, lr_a, e


def distdist_model(lr_min, m, beta1, beta2 ,lr_max, bst, za, ze, lr):

    c = numpy.zeros(lr.shape)
    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    c = m*(lr-lr_min)*( 1 / (1 + numpy.exp(  ((lr - lr_max)/(0.5*beta1))   )) )**2
    #c = m*(lr)*( 1 / (1 + numpy.exp(  ((lr - lr_max)/(0.5*beta1))   )) )**2
    #c = c - m*lr_min

    c = c + bst*numpy.exp( -(lr-(lr_max-beta1))**2/(2*beta2**2))

    c[lr > lr_cut] = 0

    c[lr < lr_min] = 0

    c[c < 0] = 0

    return c




def get_log_log_data(dists):

    if True:
        log_dists = numpy.log10(dists)
        counts, bins = numpy.histogram(log_dists, bins = 200, range=(-1, 2.7))

    bins = (bins[1:] + bins[:-1])/2
    log_counts = numpy.zeros(counts.shape)
    log_counts[counts > 0] =  numpy.log10(counts[counts > 0])
    d_counts = numpy.sqrt(counts)
    d_log_counts = numpy.zeros(d_counts.shape)
    cond = log_counts < 1e-3
    d_log_counts[~cond] = d_counts[~cond]/(counts[~cond])
    d_log_counts[cond] = 1

    return bins, log_counts, d_log_counts

def bb_distdist(x, y, dy, args):


    def lnlike_plaw(z):


        lr_min, m = z

        x, y, dy, za, ze, e = args_plaw
        lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

        fx = m*(x-lr_min)
        fx[x > lr_cut] = 0
        fx[x < lr_min] = 0

        chi2 = (fx - y)**2
        chi2 = chi2/(0.1**2 + dy**2)
        chi2 = chi2[:e]

        return numpy.nansum(chi2)

    def lnlike_all(z):



        alpha ,beta2, lr_max, bst = z
        #if bst < 0:
        #    return numpy.inf
        if bst > 1.5:
            return numpy.inf

        #if alpha < 0:
        #    return numpy.inf

        x, y, dy, za, ze, lr_min, m = args_all

        lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)


        lr_max_data = x[numpy.argmax(y)]
        if alpha < (lr_cut - lr_max_data)*0.1:
            return numpy.inf
        #if alpha > (lr_cut - lr_max_data)*4:
        #    return numpy.inf
        if alpha > 0.3:
            return numpy.inf

        if beta2 < 0:
            return numpy.inf
        #if beta2 > (lr_cut - lr_max_data)*4:
        #    return numpy.inf
        if beta2 > 0.3:
            return numpy.inf


        if lr_max < (lr_max_data - 0.01):
            return numpy.inf

        fx = distdist_model(lr_min, m, alpha, beta2 ,lr_max, bst, za, ze, x)
        chi2 = (fx - y)**2
        chi2 = chi2/(0.1**2 + dy**2)
        return numpy.nansum(chi2)
    x, y, dy, za, ze = args
    lr_min, m, alpha ,beta2, lr_max, bst, lr_a, e = initial_guess(x, y, dy, za, ze )
    #print('lrmin = {:.2f}, m = {:.2f}, b1 = {:.2f}, b2 = {:.2f}, lrmax = {:.2f}, bst = {:.2f}, lr_a = {:.2f}'.format(lr_min, m, alpha, beta2, lr_max, bst, lr_a))

    args_plaw = (x, y, dy, za, ze, e)
    res = optimize.basinhopping(lnlike_plaw, (lr_min, m), niter = 1000,)
    lr_min, m = res['x']
    args_all = (x, y, dy, za, ze, lr_min, m)
    res = optimize.basinhopping(lnlike_all, (alpha ,beta2, lr_max, bst), niter = 1000,)
    alpha ,beta2, lr_max, bst = res['x']

    res_final = (lr_min, m, alpha ,beta2, lr_max, bst)
    return res_final

def plot_res(lr_min, m, alpha, beta2, lr_max, bst, bins, log_counts, d_log_counts, za, ze, save_to=None):

    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    lr_grid = numpy.linspace(numpy.min(bins),  numpy.max(bins), 1000)
    ddm = distdist_model(lr_min, m, alpha, beta2, lr_max, bst, za, ze, lr_grid)
    plt.figure(figsize = (15,7))
    fs = 25
    lw = 7
    plt.plot(lr_grid, ddm, lw = lw, label = 'Fit, R{} = {:.2f} [Mpc], $\gamma$ = {:.2f}'.format(r'$_{\rm min}$',10**(lr_min),m), color = 'tomato', zorder=0)

    plt.errorbar(bins, log_counts, xerr = None, yerr =d_log_counts,  color = 'k', fmt='.', ms = 20, label = 'Binned MC results',capsize = 5, capthick=3, zorder=1)

    plt.vlines(x=lr_cut, ymin=-5, ymax=10, color='navajowhite' ,lw = lw+2, label='Straight line distance',zorder=2 )

    plt.title('z emission = {:.2f}, z absorption = {}'.format(ze, za), fontsize = fs)
    plt.ylim([-0.2, numpy.max(log_counts) + 0.2])
    plt.xlabel(r'log$_{10}$(R [Mpc])', fontsize = fs)
    plt.ylabel(r'log$_{10}$(number of photons)', fontsize = fs)
    plt.legend(fontsize = fs)
    if save_to:
        plt.savefig(save_to)
    plt.show()

def plot_res_res(lr_min, m, alpha, beta2, lr_max, bst, bins, log_counts, d_log_counts, za, ze, red = 0, save_to=None):

    lr_cut = numpy.log10(6 / (cosmo.h * (cosmo.Om0)**(0.5)) * (1/numpy.sqrt(1 + za) - 1/numpy.sqrt(1 + ze)) * 1000)

    lr_grid = numpy.linspace(numpy.min(bins),  numpy.max(bins), 1000)
    ddm = distdist_model(lr_min, m, alpha, beta2, lr_max, bst, za, ze, lr_grid) - red
    fig = plt.figure(figsize = (15,7))
    ax1 = fig.add_axes((.1,.3,.8,.6))
    fs = 25
    lw = 7
    plt.plot(lr_grid, ddm, lw = lw, label = 'Fit, R{} = {:.2f} [Mpc], $\gamma$ = {:.2f}'.format(r'$_{\rm min}$',10**(lr_min),m), color = 'tomato', zorder=0)

    markers, caps, bars = plt.errorbar(bins, log_counts, xerr = None, yerr =d_log_counts,  color = 'k', fmt='.', ms = 12 , label = 'Binned MC results (residuals on bottom)',capsize = 3, capthick=1, zorder=1)
    [bar.set_alpha(0.5) for bar in bars]
    [cap.set_alpha(0.5) for cap in caps]
    plt.vlines(x=lr_cut, ymin=-5, ymax=10, color='navajowhite' ,lw = lw+2, label='Straight line distance',zorder=-1 )

    plt.title('z emission = {:.2f}, z absorption = {}'.format(ze, za), fontsize = fs)
    plt.ylim([-0.2, numpy.max(log_counts) + 0.2])
    plt.xlabel(r'log$_{10}$(R [Mpc])', fontsize = fs)
    plt.ylabel(r'log$_{10}$(number of photons)', fontsize = fs)
    plt.legend(fontsize = fs-5, loc = 'upper left')
    xlim = plt.gca().get_xlim()
    ax2 = fig.add_axes((.1,.1,.8,.2))
    ddm_bins = distdist_model(lr_min, m, alpha, beta2, lr_max, bst, za, ze, bins) - red
    markers, caps, bars = plt.errorbar(bins, log_counts - ddm_bins, xerr = None, yerr =d_log_counts,  color = 'k', fmt='.', ms = 12, label = 'Residuals',capsize = 3, capthick=1, zorder=1)
    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]
    plt.grid(axis = 'y', which = 'major')
    plt.xlim(xlim)
    #plt.legend(fontsize = 20)
    #plt.ylabel('Residuals', fontsize = fs)
    plt.xlabel(r'log$_{10}$(R [Mpc])', fontsize = fs)
    plt.ylim((-0.6,0.6))
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()

def fit_all(mc_results_path):
    Lpix = 3


    ze_grid = numpy.load('{}z_source_grid.npy'.format(mc_results_path))

    dza = 0.025
    z_simul = numpy.arange(21,41)
    #z_simul = numpy.arange(35,36)

    Rshell_grid, Rmin, Rmax, nof_shells = get_Rshell_grid()

    for zcenter_ind, z_center in enumerate(z_simul):
        bst_arr = numpy.zeros(nof_shells)*numpy.nan
        m_arr = numpy.zeros(nof_shells)*numpy.nan
        lr_min_arr = numpy.zeros(nof_shells)*numpy.nan
        lr_max_arr = numpy.zeros(nof_shells)*numpy.nan
        alpha_arr = numpy.zeros(nof_shells)*numpy.nan
        beta_arr = numpy.zeros(nof_shells)*numpy.nan
        rshell_arr = numpy.zeros(nof_shells)*numpy.nan
        zshell_arr = numpy.zeros(nof_shells)*numpy.nan
        dists_arr = numpy.zeros(nof_shells)*numpy.nan
        zcenter_df = pandas.DataFrame()
        for shell_ind in trange(nof_shells):
            r_shell = Lpix*(Rmax[shell_ind] + Rmin[shell_ind])/2
            z_shell = R_to_zshell(r_shell, z_center)

            za = z_center # absorption
            ze = z_shell # emission

            dists = get_photons_za_ze(mc_results_path, za, ze, dza)
            nof_dists = len(dists)
            dists_arr[shell_ind] = nof_dists
            if nof_dists > 10000:
                try:
                    bins, log_counts, d_log_counts = get_log_log_data(dists)

                    lr_grid = numpy.linspace(numpy.min(bins), numpy.max(bins), 1000)

                    args = (bins, log_counts, d_log_counts, za, ze)
                    res = bb_distdist(bins, log_counts, d_log_counts, args)

                    lr_min, m, alpha, beta, lr_max, bst = res

                    bst_arr[shell_ind] = bst
                    m_arr[shell_ind] = m
                    lr_min_arr[shell_ind] = lr_min
                    lr_max_arr[shell_ind] = lr_max
                    alpha_arr[shell_ind] = alpha
                    beta_arr[shell_ind] = beta
                    rshell_arr[shell_ind] = r_shell
                    zshell_arr[shell_ind] = z_shell

                except:
                    print('Fail at shell {}, zcenter {}'.format(shell_ind, z_center))
        zcenter_df['m'] = m_arr
        zcenter_df['lr_min'] = lr_min_arr
        zcenter_df['lr_max'] = lr_max_arr
        zcenter_df['alpha'] = alpha_arr
        zcenter_df['beta'] = beta_arr
        zcenter_df['bst'] = bst_arr
        zcenter_df['r_shell'] = rshell_arr
        zcenter_df['z_shell'] = zshell_arr
        zcenter_df['nof_dists'] = dists_arr

        zcenter_df.to_csv('fitting_results_v3/za_dfs/za_{}_df.csv'.format(z_center))


if __name__ == '__main__':
    mc_results_path = '/Volumes/Backup/Cosmo/distance_distribution/dense_z_source_1000_nu_bins_20000_r/'
    import warnings
    warnings.filterwarnings("ignore")
    fit_all(mc_results_path)

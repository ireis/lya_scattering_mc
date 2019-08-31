import numpy
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib_params import set_plt_params
plt = set_plt_params(plt)
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from tqdm import tqdm_notebook as tqdm

from scipy.interpolate import interp2d


def bin_photons(mc_results_path):
    z_grid = numpy.load('{}z_source_grid.npy'.format(mc_results_path))

    delta = 0.05
    z_bins = numpy.arange(4, 50, 0.1) + delta
    z_bins_mean = (z_bins[1:]+z_bins[:-1])/2

    log_r_bins = numpy.log10(numpy.linspace(0.0001, 1000, 200))
    log_r_grid = (log_r_bins[1:] + log_r_bins[:-1])/2

    fraction_of_photons_observed_all_z_source = numpy.zeros([len(z_grid) ,len(log_r_bins)-1, len(z_bins)-1])

    for z_source_ind, z_source in tqdm(enumerate(z_grid)):
        log_r = numpy.load('{}log_r_z_{}.npy'.format(mc_results_path, z_source))
        z_abs = numpy.load('{}z_abs_z_{}.npy'.format(mc_results_path, z_source))

        nof_photons, _, _ = numpy.histogram2d(x = log_r, y = z_abs, bins = [log_r_bins, z_bins])

        fraction_of_photons_observed = numpy.zeros(nof_photons.shape)
        for z_idx in range(z_bins_mean.size):
            nof_photons_z = nof_photons[:, z_idx].copy()
            tot_nof_photons = numpy.sum(nof_photons_z)
            if tot_nof_photons == 0:
                fraction_of_photons_observed[:, z_idx] = 0
            else:
                fraction_of_photons_observed[:, z_idx] = nof_photons_z/tot_nof_photons


        fraction_of_photons_observed_all_z_source[z_source_ind] = fraction_of_photons_observed
    return z_bins_mean, log_r_grid, z_grid, fraction_of_photons_observed_all_z_source

def get_F(x, y, z):

    f = interp2d(x,y,z, kind='cubic')


    xmin, xmax = 0.0001, 1000
    X = numpy.log10(numpy.linspace(xmin, xmax, 900))

    ymin, ymax = 5, 50
    Y = numpy.linspace(ymin, ymax, 1000)

    F = f(X,Y)

    F[F < 0.001] = 0
    nF = F.shape[1]
    mF = F.shape[0]
    for i in range(mF):
        if numpy.sum(F[i]) > 1:
            to_z = False
            m = numpy.argmax(F[i])
            for j in range(m, nF):
                if to_z:
                    F[i,j] = 0
                if F[i,j] == 0:
                    to_z = True
            to_z = False
            for j in range(m, 0, -1):
                if to_z:
                    F[i,j] = 0
                if F[i,j] == 0:
                    to_z = True

            F[i] = F[i]/numpy.sum(F[i])

        else:
            F[i] = 0
    return X,Y,F


def get_photon_fraction(mc_results_path, output_path):

    binned_nof_photons_path = '{}binned_data.npy'.format(mc_results_path)
    try:
        z_bins_mean, log_r_grid, z_grid, fraction_of_photons_observed_all_z_source = numpy.load(binned_nof_photons_path)
    except:
        z_bins_mean, log_r_grid, z_grid, fraction_of_photons_observed_all_z_source = bin_photons(mc_results_path)
        numpy.save(binned_nof_photons_path, [z_bins_mean, log_r_grid, z_grid, fraction_of_photons_observed_all_z_source])

    lya_scattered_photon_r_dist = numpy.zeros([fraction_of_photons_observed_all_z_source.shape[0], fraction_of_photons_observed_all_z_source.shape[1]])
    z_simul = numpy.arange(5,50)
    for z in tqdm(z_simul):
        z_center_idx = numpy.argmin(abs(z - z_bins_mean))
        z_center = z_bins_mean[z_center_idx]
        lya_scattered_photon_r_dist = fraction_of_photons_observed_all_z_source[:,:,z_center_idx]
        X,Y, F = get_F(log_r_grid, z_grid, lya_scattered_photon_r_dist.copy())
        mdict = {'photon_fractions': F}
        savemat('{}lya_scattering_{}.mat'.format(output_path,z), mdict)

    mdict = {'R_lya_scatter': 10**X}
    savemat('{}lya_scattering_r_grid.mat'.format(output_path), mdict)
    mdict = {'z_lya_scatter': Y}
    savemat('{}lya_scattering_z_grid.mat'.format(output_path), mdict)

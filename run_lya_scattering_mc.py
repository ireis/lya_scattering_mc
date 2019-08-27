import lya_scattering_mc_fast as lya
import numpy
from joblib import Parallel, delayed




def single_z_source(z, nof_nu_bins, nof_R):
    print('At z = {}'.format(z))
    lya_scatter = lya.LyaSctr_MC(z_s=z, nof_nu_bins=nof_nu_bins, nof_R=nof_R)
    lya_scatter.run()
    z_abs = lya_scatter.absorption_zs.flatten()
    log_r = numpy.log10(lya_scatter.R_all.value.flatten())

    fname_log_r = '{}/log_r_z_{}'.format(save_to, z)
    fname_z_abs = '{}/z_abs_z_{}'.format(save_to, z)

    numpy.save(fname_log_r, log_r)
    numpy.save(fname_z_abs, z_abs)
    return


# The code returns -nof_R- radii in each of -nof_nu_bins- nu bins (the bins are between lyman alpha and lyman beta)
# for a given redshift.
if __name__ = '__main__':
    nof_nu_bins = 5000
    nof_R = 5000
    save_to = '{}_nu_bins'.format(nof_nu_bins)
    z_source_grid = numpy.arange(5,50)
    numpy.save('{}/z_source_grid'.format(save_to), z_source_grid)
    Parallel(n_jobs=-1, verbose=10)(delayed(single_z_source)(z, nof_nu_bins, nof_R) for z in z_source_grid)


#for z in z_source_grid:
#    single_z_source(z)

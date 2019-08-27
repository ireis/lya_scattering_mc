import lya_scattering_mc as lya
import numpy
from joblib import Parallel, delayed


z_source_grid = numpy.arange(5,50)
numpy.save('5000_nu_bins/z_source_grid', z_source_grid)

def single_z_source(z):
    print('At z = {}'.format(z))
    lya_scatter = lya.LyaSctr_MC(z_s=z, nof_nu_bins=5000, nof_R=2000)
    lya_scatter.run()
    z_abs = lya_scatter.absorption_zs.flatten()
    log_r = numpy.log10(lya_scatter.R_all.value.flatten())

    fname_log_r = '5000_nu_bins/log_r_z_{}'.format(z)
    fname_z_abs = '5000_nu_bins/z_abs_z_{}'.format(z)

    numpy.save(fname_log_r, log_r)
    numpy.save(fname_z_abs, z_abs)
    return


# The code returns -nof_R- radii in each of -nof_nu_bins- nu bins (the bins are between lyman alpha and lyman beta)
# for a given redshift.

#Parallel(n_jobs=1)(delayed(single_z_source)(z) for z in z_source_grid)

for z in z_source_grid:
    single_z_source(z)

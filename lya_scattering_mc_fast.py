from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
import numpy
from numba import njit, prange
#from tqdm import  tqdm
#from tqdm import tqdm_notebook as tqdm

global nu_a, h, Om0, Ob0
nu_a = 2.45 * (10**15) # * u.Hertz
h = cosmo.h
Om0 = cosmo.Om0
Ob0 = cosmo.Ob0

class LyaSctr_MC():
    def __init__(self, z_s, nof_R = 1000, nof_nu_bins = 10):
        self.nu_a        = nu_a * u.Hertz
        self.nu_b        = self.nu_a * (32/27)
        self.z_s         = z_s
        self.nu_star_z_s = calc_nu_star(z_s) * u.Hertz
        self.nu_min      = self.nu_a + 0.1*self.nu_star_z_s
        if type(nof_R) == int:
            self.nof_R       = [nof_R]*nof_nu_bins
        elif type(nof_R) == list:
            if len(nof_R) == nof_nu_bins:
                self.nof_R = nof_R
            else:
                print('nof_R length should match nof_nu_bins, using 1000 Rs for each source redshift')
                self.nof_R = [1000]*nof_nu_bins
        else:
            print('nof_R should be either an integer or a list (with length = nof_nu_bins), using 1000 Rs for each source redshift')
            self.nof_R = [1000]*nof_nu_bins

        self.max_nof_R = numpy.max(self.nof_R)
        self.R_all       = -numpy.ones([nof_nu_bins,self.max_nof_R ]) * u.Mpc
        self.absorption_zs = -numpy.ones([nof_nu_bins,self.max_nof_R ])
        self.z           = z_s
        self.nof_nu_bins = nof_nu_bins


    def get_nu_bins(self):
        dnu_min = self.calc_delta_nu(self.nu_min)
        dnu_max = self.calc_delta_nu(self.nu_b)
        dnu_max = self.calc_delta_nu(4 * (10**15) * u.Hertz)

        log_dnu_min = numpy.log(dnu_min)
        log_dnu_max = numpy.log(dnu_max)

        log_nu_grid = numpy.linspace(log_dnu_min, log_dnu_max, self.nof_nu_bins + 1)
        log_nu_1 = log_nu_grid[:-1]
        log_nu_2 = log_nu_grid[1:]

        nu1 = (numpy.exp(log_nu_1) + 1)*self.nu_a
        nu2 = (numpy.exp(log_nu_2) + 1)*self.nu_a

        nu = numpy.sqrt(nu1*nu2)

        self.nu_grid  = nu
        self.nu1_grid = nu1
        self.nu2_grid = nu2

        return


    def calc_delta_nu(self, nu):
        return (nu/self.nu_a - 1)




    def run(self):

        self.get_nu_bins()


        for bin_idx in range(self.nof_nu_bins):

            self.nu_s   = self.nu_grid[bin_idx]
            self.tau_f1 = (self.nu_star_z_s/self.nu_a) * (self.nu_s / self.nu_a)**(3/2)


            v2_fill = self.nu1_grid[bin_idx]

            curr_nof_R = self.nof_R[bin_idx]
            global_seed = numpy.random.randint(10000)

            R_bin_vals = numpy.zeros(curr_nof_R)

            z_obs = get_R_for_bin(R_bin_vals, self.R_all.value, bin_idx, self.z_s, self.nu_s.value, self.nu_min.value, self.tau_f1, self.nu2_grid.value, curr_nof_R, v2_fill.value, global_seed)

            self.absorption_zs[bin_idx,:] = z_obs
            self.R_all[bin_idx] = R_bin_vals*u.Mpc

            final_itr = (bin_idx == (self.nof_nu_bins-1))
            if not final_itr:
                self.R_all[bin_idx+1,:] = self.R_all[bin_idx,:].copy()



        return

@njit
def get_R_for_bin(R_bin, R_all, bin_idx, z_s, nu_s, nu_min, tau_f1, nu2_grid, curr_nof_R, v2_fill, global_seed):
    seed = global_seed

    z_obs  = (1 + z_s)*nu_a/nu_s - 1

    for R_idx in range(curr_nof_R):
        numpy.random.seed(seed)
        seed = seed + 1

        #initialize a new photon at z source, with R = 0 and the bin frequency
        R  = 0
        z  = z_s
        nu = nu_s

        high_tau = nu < nu_min
        next_bin = nu < v2_fill

        while (not high_tau) and (not next_bin):
            # ietrative scattering, nu, R and z are updated in each iteration
            nu, z, R = low_tau_sctr(nu, z, R, z_obs, tau_f1)

            # check if the photon reached the end of its path
            # or swithced to the frequency of the next/prev bin
            high_tau = nu < nu_min
            next_bin = nu < v2_fill


        if high_tau:
            R_bin[R_idx] = high_tau_sctr(nu, z, R)
        elif next_bin:
            curr_bin = numpy.min(numpy.where(nu < nu2_grid)[0])
            R_to_end = numpy.random.choice(R_all[curr_bin,:]) # *u.Mpc
            R_bin[R_idx] = calc_full_R(R_to_end, R)
    return z_obs


@njit
def low_tau_sctr(nu, z, R, z_obs, tau_f1):


    U2 = numpy.random.rand()
    tau = - numpy.log(U2)

    z_grid = numpy.linspace(z, z_obs-0.0000001, 500)
    new_z = z_grid[numpy.argmin(numpy.abs(tau - calc_tau(z_grid, nu, z, tau_f1)))]

    # New frequency
    nu = redshifted_nu(nu, z, new_z)

    # New distance
    L = comove_length(z, new_z)
    U1 = numpy.random.rand()
    mu = 2*U1 - 1
    R  = (R**2 + L**2 + 2*R*L*mu)**0.5

    #New redshift
    z  = new_z

    return nu, z, R

@njit
def calc_tau(z_grid, nu, z, tau_f1):

    f1 = tau_f1

    arg1 = (nu/nu_a) -1

    up   = nu  *(1 +  z_grid)
    down = nu_a*(1 +  z)
    arg2 = (up/down - 1)

    f2 = calc_F(arg1) - calc_F(arg2)

    tau = f1*f2

    return tau

@njit
def calc_F(y):

    up   = numpy.sqrt(1+y) + 1
    down = numpy.sqrt(1+y) - 1
    log_arg = numpy.abs(up/down)
    term1 = (5/2)*numpy.log(log_arg)

    up   = 5*y**2 + (20/3)*y + 1
    down = y*(y+1)**(3/2)
    term2 = up/down

    F = term1 - term2

    return F

@njit
def redshifted_nu(nu_0, z_0, z_f):
    nu_f = nu_0 * (1 + z_f) / (1 + z_0)
    return nu_f

@njit
def comove_length(z_0, z_f):

    up = 6.0 * 1000 # Gpc to Mpc
    down = (Om0**0.5) * h
    f1 = up/down

    a_f   = 1 / (1 + z_f)
    a_0 = 1 / (1 + z_0)
    f2 = numpy.sqrt(a_f) - numpy.sqrt(a_0)

    L = f1*f2

    return L

@njit
def high_tau_sctr(nu, z, R):

    z_obs = z # right? or self.z_obs
    a_obs = 1 / (1 + z_obs)

    up   = nu - nu_a
    down = calc_nu_star(z_obs)
    nu_tild = up/down


    up   = 2*(nu_tild**3)
    down = 9*(z_obs ** 2)
    var  = up/down
    Ls   = numpy.random.normal(loc = 0, scale = var, size = 3) # Lx, Ly, Lz

    L_tild = numpy.sqrt(numpy.sum(Ls**2))

    R_star = calc_R_star(z_obs)

    L = L_tild*R_star

    U1 = numpy.random.rand()
    mu = 2*U1 - 1

    return (R**2 + L**2 + 2*R*L*mu)**0.5

@njit
def calc_nu_star(z):

    f1 = 5.58*(10**12) # u.Hertz
    f2 = Ob0 * h * (Om0)**(-0.5)
    f3 = (1 + z)**(3/2)

    nu_star = f1*f2*f3

    return nu_star

@njit
def calc_R_star(z):

    f1 = 6.77 #* u.Mpc
    f2 = Om0/Ob0
    f3 = 1 + z

    R_star = f1*f2*f3

    return R_star

@njit
def calc_full_R(L, R):

    U1 = numpy.random.rand()
    mu = 2*U1 - 1

    return (R**2 + L**2 + 2*R*L*mu)**0.5

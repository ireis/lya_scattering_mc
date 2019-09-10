from tqdm import trange
import numpy
from numba import njit
from scipy.io import savemat
import json

def get_cosmo():
    f = open("cosmo.json","r")
    cosmo = json.load(f)
    f.close()
    return cosmo

cosmo = get_cosmo()
k = cosmo['k']
H0 = cosmo['H0']
Om = cosmo['Om']
OLambda = cosmo['OLambda']

R_horizon_f = 5.05*(cosmo['Om'] * cosmo['h']**2 / 0.141)**(-0.5)* 1000




def get_horizon(z, R_horizon_f=R_horizon_f):
    '''
    The particle horizon in Mpc
    '''
    R = R_horizon_f * ( (1 + z)/10 )**(-0.5)
    return R


def get_Rshell_grid():
    '''
    The grid of shell Radii for the sum over spherical shells calculation
    '''

    Rshell_grid = numpy.concatenate([numpy.linspace(1e-10,        float(500)/3,   110),
                                     numpy.logspace(numpy.log10(float(550)/3), numpy.log10(float(15000)/3), 20 )])

    Rmin = Rshell_grid[:-1]
    Rmax = Rshell_grid[1:]

    nof_shells = len(Rmin);

    return Rshell_grid, Rmin, Rmax, nof_shells

def R_to_zshell(R, z, k=k):
    '''
    The redshift zshell from which radiation that tracelled a distance R arrived ot redshift z
    '''

    if (R < get_horizon(z)):
        zshell = (k**2*z + 2*k*R*numpy.sqrt(1+z) - R**2*(1+z))/(k**2 - 2*k*R*numpy.sqrt(1+z) + R**2*(1+z));
    else:
        zshell = 1e15

    return zshell





def get_r_zaze(p, za, ze):
    za_ind = numpy.argmin(abs(za - za_grid))
    ze_ind = numpy.argmin(abs(ze - ze_grid))
    za = za_grid[za_ind]
    ze = ze_grid[ze_ind]

    r_zaze = numpy.load('{}za_{}_ze_{}.npy'.format(p, za_ind, ze_ind))
    return za, ze, r_zaze



@njit
def get_fraction_of_photons(r, r_zaze):
    loc = numpy.argmin(numpy.abs(r_grid - r))
    r_for_v = r_grid[loc]
    fop = r_zaze[loc]
    fop_per_v = fop/r_for_v**2
    return fop_per_v

@njit
def get_real_space_window(real_space_window, r_zaze, N):
    mid = int(N/2) + 1
    for xi in range(mid):
        for yi in range(mid):
            for zi in range(mid):
                if (xi >= yi) and (yi >= zi):
                    res = get_fraction_of_photons(r_cube[xi,yi,zi], r_zaze)

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

    #norm = numpy.sum(real_space_window)
    #real_space_window = real_space_window/norm

    return #norm






if __name__ == '__main__':

    p_window = '/scratch300/itamarreis/git/poisson_21cm/IC_and_backgrounds/lya_window_functions_py/'
    z_simul = numpy.arange(5,41)
    Rshell_grid, Rmin, Rmax, nof_shells = get_Rshell_grid()


    global r_grid, r_cube

    N = 128
    mid_cube = float(N)/2
    Lpix = 3
    real_space_window = numpy.zeros([N,N,N])


    x = numpy.arange(N) - mid_cube
    y = numpy.arange(N) - mid_cube
    z = numpy.arange(N) - mid_cube

    r_cube = numpy.sqrt(x[:,None,None]**2 + y[None,:,None]**2 + z[None,None,:]**2)*Lpix






    for z_center in z_simul:
        za_ind = numpy.argmin(abs(z_center - za_grid))
        za_ = za_grid[za_ind]
        print(z_center)
        for ri in trange(nof_shells):
            r_shell = Lpix*(Rmax[ri] + Rmin[ri])/2
            z_shell = R_to_zshell(r_shell, z_center)
            za_, ze_, r_zaze = get_r_zaze(p, z_center, z_shell)
            real_space_window = numpy.zeros([N,N,N])
            get_real_space_window(real_space_window, r_zaze, N)
            real_space_window = real_space_window/numpy.sum(real_space_window)
            shifted_real_space_window = numpy.fft.ifftshift(real_space_window)
            k_space_window = numpy.fft.fftn(shifted_real_space_window)
            m_dict = {'windowk':k_space_window}
            savemat('{}za_{}_R_{:.4f}.mat'.format(p_window, za, r_shell), m_dict)

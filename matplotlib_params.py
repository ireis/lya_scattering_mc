
def set_plt_params(plt):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.labelsize'] = 'large'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['xtick.labelsize'] = 'large'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = 15, 5
    return plt

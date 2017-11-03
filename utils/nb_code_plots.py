# -------- Copy this into your ipython notebook and run -------- #
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider

# average shots
def inter_avg_shot_plot(dset, x = x_i, color = list(plot_colors.keys()), Error = True):
    plt.figure()
    plt.title("Shots averaged at {} cm".format(x))
    avg_shot_plot(dset, x, np.arange(n_shots_per_x), color, error = Error)
    plt.show()

# time sliders setup
t_i_widget = IntSlider(min = 0, max = len(dset['signal'][0]), value = 0)
t_f_widget = IntSlider(min = 1, max = len(dset['signal'][0]), value = 10000)
def update_t_range(*args):
    t_i_widget.max = t_f_widget.value - 1
t_f_widget.observe(update_t_range, 'value')

# profiles
def inter_profile_plot(dset, t_i, t_f, color = list(plot_colors.keys()), Error = False): 
    plt.figure()
    plt.title("Profile averaged from frame (ms) {} ({}) to {} ({})".format(t_i, t_i / 6.25e3, t_f, t_f / 6.25e3))
    profile_plot(dset, np.arange(t_i, t_f), np.arange(n_shots_per_x), color, Error)
    plt.show()

def inter_plot_fluctuations(dset, t_i, t_f, x = x_i):
	plt.figure()
	plt.title("RMS fluctuations from frame (ms) {} ({}) to {} ({}) at {} cm".format(t_i, t_i / 6.25e3, t_f, t_f / 6.25e3, x))
	plot_fluctuations(dset, np.arange(t_i, t_f), np.arange(n_shots_per_x), x)
	plt.show()

def inter_plot_power_spectrum(dset, t_i, t_f, x = x_i):
	plt.figure()
	plt.title("Power spectrum from frame (ms) {} ({}) to {} ({}) at {} cm".format(t_i, t_i / 6.25e3, t_f, t_f / 6.25e3, x))
	plot_power_spectrum(dset, np.arange(t_i, t_f), np.arange(n_shots_per_x), x)
	plt.show()

# -------- Interact with plots -------- #
# would recommend continuous_update = False if not using interact_manual
interact_manual(inter_avg_shot_plot, dset = fixed(dset), x = (x_i, x_f, dx))
interact_manual(inter_profile_plot, dset = fixed(dset), t_i = t_i_widget, t_f = t_f_widget)
interact_manual(inter_plot_fluctuations, dset = fixed(dset), t_i = t_i_widget, t_f = t_f_widget, x = (x_i, x_f, dx))
interact_manual(inter_plot_power_spectrum, dset = fixed(dset), t_i = t_i_widget, t_f = t_f_widget, x = (x_i, x_f, dx))
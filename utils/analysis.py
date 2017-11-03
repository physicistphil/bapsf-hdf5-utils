import matplotlib.pyplot as plt
import numpy as np


# -------- Constants -------- #
plot_colors = {"red": ("#CC4F1B", "#FF9848"), "green": ("#3F7F4C", "#7EFF99"), "blue": ("#1B2ACC", "#089FFF")}


# -------- Data run parameters -- make sure to change this every time you handle a new dataset -------- #
board, channel = 1, 3
probe_order = 0 # count from 0
n_shots_per_x = 26 
x_i = 5
x_f = 45
nx = 81

num_shots = n_shots_per_x * nx
dx = (x_f - x_i) / (nx - 1)


# -------- Useful functions -------- #

def calc_index (dset, position, shot = 0): 
	index = np.matrix(np.rint((position - x_i) / dx * n_shots_per_x + 
	probe_order * num_shots)).astype(int) + np.matrix(shot).T
	return index

def shot_plot (dset, position, shot = 0):
	index = np.asarray(calc_index(dset, position, shot)).flatten()[0]
	plt.plot(dset['signal'][index])

def avg_shot_plot (dset, position, shots, color = 'blue', error = False): 
	index = calc_index(dset, position, shots)
	average = np.mean(dset['signal'][index], axis = 0).squeeze()
	plt.plot(average, color = plot_colors[color][0], linewidth = 1)

	if error:
		std_dev = np.std(dset['signal'][index], axis = 0).squeeze()
		plt.fill_between(np.arange(len(average)), average - std_dev, average + std_dev, alpha = 0.2, 
		edgecolor = plot_colors[color][0], facecolor = plot_colors[color][1], linewidth = 0.5, 
			antialiased = True)

# time and shot need to be np.arange(something)
def profile_plot (dset, time = np.array([0]), shot = np.arange(n_shots_per_x), color = 'blue', error = False):
	if type(time) != np.ndarray: time = np.array([time])
	if type(shot) != np.ndarray: time = np.array([shot])
	index = calc_index(dset, np.linspace(x_i, x_f, nx), shot)

	# average selected shots (from the index) and times at each position
	mean = np.mean(dset['signal'][index][:, :, time], axis = (0, 2))
	actual_radius = np.linspace(x_i, x_f, nx)
	plt.plot(actual_radius, mean, color = plot_colors[color][0], linewidth = 1)

	if error: 
		# compute the whole standard deviation of the flattened array along shots and times
		std_dev = np.std(dset['signal'][index][:, :, time], axis = (0,2))
		plt.fill_between(actual_radius, mean - std_dev, mean + std_dev, alpha = 0.2, 
		edgecolor = plot_colors[color][0], facecolor = plot_colors[color][1], linewidth = 0.5, 
			antialiased = True)

def calc_rms_fluctuations (dset, time = np.array([0]), shot = np.array([0]), positions = np.array([x_i])):
	if type(time) != np.ndarray: time = np.array([time])
	if type(shot) != np.ndarray: shot = np.array([shot])
	if type(positions) != np.ndarray: positions = np.array([positions])

	index = calc_index(dset, positions, shot)
	mean = np.mean(dset['signal'][index][:, :, time], axis = 0)
	fluct = np.swapaxes(dset['signal'][index][:, :, time], 0, 1) - mean[:, np.newaxis]
	fluct_rms = np.sqrt(np.mean(fluct ** 2, axis = 1))
	return fluct, fluct_rms

def plot_fluctuations (dset, time = np.array([0]), shot = np.array([0]), positions = np.array([x_i])):
	fluct, fluct_rms = calc_rms_fluctuations(dset, time, shot, positions)
	for i in fluct_rms: plt.plot(i)

def plot_power_spectrum (dset, time = np.array([0]), shot = np.array([0]), positions = np.array([x_i])):
	fluct, fluct_rms = calc_rms_fluctuations(dset, time, shot, positions)
	fluct_power_spectrum = np.abs(np.fft.rfft(fluct_rms, axis = 1)) ** 2
	for i in fluct_power_spectrum: plt.loglog(i)
	

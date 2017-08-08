import h5py
import hdf5_basics
import h5Parse
import matplotlib.pyplot as plt
import numpy as np
import math

class shot_data:
	def __init__ (self, ** kwargs):
		if len(kwargs) == 0:
			print ("No file path given. Load data with <class instance>.openFile(filename = \'path\')")
			return
		else: 
			self.openFile(kwargs['filename'])
			return

	def openFile (self, filename):
		if type(filename) is not str:
			print ("Filename needs to be a string")
			return

		self.__f = h5py.File(filename)
		self.__data_struct = h5Parse.openHDF5_dataset(self.__f)
		self.clock_rate = self.__data_struct['clock rate']
		self.data_type = self.__data_struct['data type'].decode['UTF8']
		self.digitizer = self.__data_struct['digitizer']
		self.info = self.__data_struct['data'].name
		self.file = filename
		self.data = self.__data_struct['data']

# TODO: 
# 	include shot count per position
#	include total number of positions 
		


# the code in here has been optimized to only load the data from disk that we need
def calc_index (position, probe_order, num_shots = 25, num_positions = 91):
	data_position = int(round((45.0 - position) * 2)) 	# convert to data coordinates
	index = num_positions * num_shots * probe_order + data_position * num_shots
	return index

def one_shot_plot (data, index, shot = 0):
	plot_data = bit_to_voltage(data[index + shot])
	plt.plot(hdf5_basics.bit_to_voltage(data[index + shot]))
	# plt.show()
	return

def multi_shot_plot (data, index, num_shots = 25):
	x, y = int(math.ceil(math.sqrt(num_shots))), int(math.floor(math.sqrt(num_shots)))
	plots, plotarr = plt.subplots(x ,y)
	for i in range(num_shots):
		plotarr[int(i // x)][int(i % y)].plot(hdf5_basics.bit_to_voltage(data[index + i]))
	# plots.show()
	return 

def avg_shot_plot (data, index, num_shots = 25):
	subarray = hdf5_basics.bit_to_voltage(data[index : index + num_shots])
	avg_array = np.mean(subarray, axis = 0) 
	dev_array = np.std(subarray, axis = 0)

	plt.figure()
	plt.title("Plot averaged over {} shots".format(num_shots))
	plt.plot(range(len(avg_array)), avg_array, color = "#1B2ACC")
	plt.fill_between(range(len(avg_array)), avg_array - dev_array, avg_array + dev_array, alpha = 0.2, 
		edgecolor = "#1B2ACC", facecolor = "#089FFF", linewidth = 1, antialiased = True)
	return

def get_profile (data, index, time_i = 0, time_f = 1, num_shots = 25, num_positions = 91):
	# slice only the data we want (doesn't hit the disk too hard)
	sliced_data = data[index - num_shots * (num_positions -1) : index + num_shots, time_i : time_f]
	# reshape so it's easier to work with (and convert to voltage from bits)
	data_cube = hdf5_basics.bit_to_voltage(sliced_data.reshape(91, 25, time_f - time_i))
	avg_data = np.mean(data_cube, axis = 1)
	dev_data = np.std(data_cube, axis = 1)

	return avg_data, dev_data

def profile_plot (data, index, time = 0, num_shots = 25, num_positions = 91):
	avg_data, dev_data = get_profile(data, index, time, time + 1, num_shots, num_positions)
	
	plt.figure()
	plt.title("Profile averaged over {} shots at time = {}".format(num_shots, time))
	plt.plot(range(len(avg_data[:,0])), avg_data[:,0], color = "#1B2ACC")
	plt.fill_between(range(len(avg_data[:,0])), avg_data[:,0] - dev_data[:,0], avg_data[:,0] + dev_data[:,0], 
		alpha = 0.2, edgecolor = "#1B2ACC", facecolor = "#089FFF", linewidth = 1, antialiased = True)

	# plt.plot(avg_data)
	return 

# time_i and time_f are inclusive: [time_i, time_f]. Note: no sanity checks
def avg_profile_plot (data, index, time_i, time_f, num_shots = 25, num_positions = 91):
	avg_data, dev_data = get_profile(data, index, time_i, time_f + 1, num_shots, num_positions)
	
	# find the average and standard deviation of the whole set 
	mean = np.mean(avg_data, axis = 1)
	var = dev_data ** 2
	mean_var = np.mean(var, axis = 1)
	stddev = mean_var ** 0.5

	plt.figure()
	plt.title("Profile {} shot-averaged, time-averaged over t = {} to {}".format(num_shots, time_i, time_f))
	plt.plot(range(len(mean)), mean, color = "#CC4F1B")
	plt.fill_between(range(len(mean)), mean - stddev, mean + stddev, 
		alpha = 0.2, edgecolor = "#CC4F1B", facecolor = "#FF9848", linewidth = 1, antialiased = True)

	return


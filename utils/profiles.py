import h5py
import hdf5_basics
import h5Parse
import matplotlib.pyplot as plt
import numpy as np
import math

class probe_data:
	def __init__ (self, ** kwargs):
		if len(kwargs) == 0:
			print ("No file path given. Load data with <class instance>.openFile(filename = \'path\')")
			return
		else: 
			self.open_file(kwargs['filename'])
			return

	def open_file (self, filename):
		if type(filename) is not str:
			print ("Filename needs to be a string")
			return

		self.__f = h5py.File(filename)
		self.__data_struct = h5Parse.openHDF5_dataset(self.__f)
		self.clock_rate = self.__data_struct['clock rate']
		self.data_type = self.__data_struct['data type'].decode('UTF8')
		self.digitizer = self.__data_struct['digitizer']
		self.info = self.__data_struct['data'].name
		self.file = filename
		self.data = self.__data_struct['data']

		self.set_num_shots_positions();

	# we may want to changes this if we have 2d shot data instead of 1d
	def set_num_shots_positions (self):
		self.num_shots = int(input("Number of shots per position: "))
		self.__first_position = float(input("First position (in cm): "))
		self.__last_position = float(input("Last position (in cm): "))
		self.__position_step = float(input("Step between positions (in cm): "))
		self.num_positions = int(round(((self.__last_position - self.__first_position) / self.__position_step) + 1))
		self.probe_order = int(input("Enter probe order (starting at 0): "))

	# calculate the index at a certain position (in cm)
	def calc_index (self, position):
		data_position = int(round((45 - position) * 2))
		index = self.num_positions * self.num_shots * self.probe_order + data_position * self.num_shots
		return index

	# def get_real_time (self, frame):
		# return frame / self.clock_rate

	def get_real_time_micro (self, frame):
		return frame / self.clock_rate / 1e-6

	def get_movement_range (self):
		return np.linspace(self.__first_position, self.__last_position, num = self.num_positions)

	def one_shot_plot (self, position, shot):
		plot_data = self.data[self.calc_index(position) + shot]
		plt.title("{} shot number {} at {} cm".format(self.data_type, shot, position))
		plt.xlabel('Time (μs)')
		plt.ylabel('Voltage (V)')
		plt.plot(self.get_real_time_micro(np.arange(len(plot_data))), hdf5_basics.bit_to_voltage(plot_data))

	def avg_shot_plot (self, position):
		subarray = hdf5_basics.bit_to_voltage(self.data[self.calc_index(position) : self.calc_index(position) 
			+ self.num_shots])
		avg_array = np.mean(subarray, axis = 0) 
		dev_array = np.std(subarray, axis = 0)
		# plt.figure()
		plt.title("{} at {} cm averaged over {} shots".format(self.data_type, position, self.num_shots))
		plt.xlabel('Time (μs)')
		plt.ylabel('Voltage (V)')
		plt.plot(self.get_real_time_micro(np.arange(len(avg_array))), avg_array, color = "#1B2ACC")
		plt.fill_between(self.get_real_time_micro(np.arange(len(avg_array))), avg_array - dev_array, avg_array + 
			dev_array, alpha = 0.2, edgecolor = "#1B2ACC", facecolor = "#089FFF", linewidth = 1, antialiased = True)

	# time is in μs
	def get_profile (self, time_i, time_f):
		index = self.calc_index(0)
		time_i = int(round(time_i * self.clock_rate * 1e-6))	# convert to frame number
		time_f = int(round(time_f * self.clock_rate * 1e-6))

		# slice only the data we want (doesn't hit the disk too hard)
		sliced_data = self.data[index - self.num_shots * (self.num_positions - 1) : index + self.num_shots, 
			time_i : time_f]
		# reshape so it's easier to work with (and convert to voltage from bits)
		data_cube = hdf5_basics.bit_to_voltage(sliced_data.reshape(91, 25, time_f - time_i))
		avg_data = np.mean(data_cube, axis = 1)
		dev_data = np.std(data_cube, axis = 1)
		return avg_data, dev_data

	# time is in μs
	def profile_plot (self, time = 0):
		avg_data, dev_data = self.get_profile(time, time + 1)
		# plt.figure()
		plt.title("{} profile averaged over {} shots at {} μs".format(self.data_type, self.num_shots, time))
		plt.xlabel('Radius (cm)')
		plt.ylabel('Voltage (V)')
		plt.plot(self.get_movement_range(), avg_data[:, 0], color = "#1B2ACC")
		plt.fill_between(self.get_movement_range(), avg_data[:,0] - dev_data[:,0], avg_data[:,0] + dev_data[:,0], 
			alpha = 0.2, edgecolor = "#1B2ACC", facecolor = "#089FFF", linewidth = 1, antialiased = True)

	# time is in μs
	def avg_profile_plot (self, time_i, time_f):
		avg_data, dev_data = self.get_profile(time_i, time_f)
		mean = np.mean(avg_data, axis = 1)
		var = dev_data ** 2
		mean_var = np.mean(var, axis = 1)
		stddev = mean_var ** 0.5

		# plt.figure()
		plt.title("{} profile".format(self.data_type))
		# plt.title("{} profile averaged over {} shots from {} μs to {} μs".format(self.data_type, self.num_shots, 
			# time_i, time_f))
		plt.xlabel('Radius (cm)')
		plt.ylabel('Voltage (V)')
		plt.plot(self.get_movement_range(), mean, color = "#1B2ACC")
		plt.fill_between(self.get_movement_range(), mean - stddev, mean + stddev, 
			alpha = 0.2, edgecolor = "#1B2ACC", facecolor = "#089FFF", linewidth = 1, antialiased = True)
		# show 2 * stddev area
		# plt.fill_between(self.get_movement_range(), mean - stddev * 2, mean + stddev * 2, 
			# alpha = 0.1, edgecolor = "#1B2ACC", facecolor = "#A0CFFF", linewidth = 1, antialiased = True)
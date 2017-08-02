import h5py
import hdf5_basics
import matplotlib.pyplot as plt
import numpy as np
import math

def calc_index (position, probe_order, num_shots = 25, num_positions = 91):
	data_position = int(round((45.0 - position) * 2)) 	# convert to data coordinates
	index = num_positions * num_shots * probe_order + data_position * num_shots
	return index

def one_shot_plot (gen_data, index, shot = 0):
	plt.plot(gen_data[index + shot][3])
	# plt.show()
	return

def multi_shot_plot (gen_data, index, num_shots = 25):
	x, y = int(math.ceil(math.sqrt(num_shots))), int(math.floor(math.sqrt(num_shots)))
	plots, plotarr = plt.subplots(x ,y)
	for i in range(num_shots):
		plotarr[i / x][i % y].plot(gen_data[index + i][3])
	# plots.show()
	return 

def avg_shot_plot (gen_data, index, num_shots = 25):
	arrays = ([item for sublist in gen_data[index : index + num_shots] for item in sublist])[3 : : 4]
	avg_array = np.mean(np.array([i for i in arrays]), axis = 0) 
	plt.plot(avg_array)
	# plt.show()
	return

def get_profile (gen_data, index, time = 0, num_shots = 25):
	profile_arr = []
	for pos in np.linspace(0.0, 45.0, 91): 
		index = calc_index (pos, 1)
		arrays = ([item for sublist in gen_data[index : index + num_shots] for item in sublist])[3 : : 4]
		profile_arr.append(np.mean(np.array([i for i in arrays]), axis = 0)[time])
	return profile_arr

def profile_plot (gen_data, index, time = 0, num_shots = 25):
	profile_arr = get_profile(gen_data, index, time, num_shots)
	plt.plot(profile_arr)
	return 

# def profile_animation ():

	






# parameters
'''
position = 0 	# in cm. 0 is the very center, 45 is outtermost position
probe_order = 1 	# count from 0. 1 = 2nd to go
num_shots = 25 		# the number of shots for each position 
num_positions = 91	# the total number of positions a probe can be in
index = calc_index(position, probe_order, num_shots, num_positions);
'''

'''
f = h5py.File('/data/BAPSF_Data/Multi_Ion_Transport/July17/01_RadLine_HotEmissive.hdf5')
gen_data = hdf5_basics.generalized_data(f, 25)
'''

# flatten
'''
sliced = gen_data[index : index + 25]
flat_slice = [item for sublist in sliced for item in sublist]
flat_slice[3::4]  # gets you all the arrays
'''

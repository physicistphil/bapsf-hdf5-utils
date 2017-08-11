import hdf5_basics
import h5py
import profiles
import matplotlib.pyplot as plt
import matplotlib.animation as animation

f = h5py.File('/data/BAPSF_Data/Multi_Ion_Transport/July17/01_RadLine_HotEmissive.hdf5')
gen_data = hdf5_basics.generalized_data(f, 25)

time_arr = []
for i in range(0, 100352, 500):                                           
	time_arr.append(profiles.get_profile(gen_data, profiles.calc_index(0, 1), i))

fig, ax = plt.subplots()
xdata, yadata = [], []
line, = plt.plot([], [], 'b-', animated = True)
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
def init ():
	ax.set_xlim(0, 91)
	ax.set_ylim(0, 0.40)
	time_text.set_text('time = ')
	return line,

def func (i):
	time_text.set_text('time = %.1d' % i)
	line.set_data(list(range(91)), time_arr[i])
	# time_text.set_text('time = %.1d' % i * 500.0 * 8.0e-5)
	return line, time_text

plt.xlim(0, 91)
plt.ylim(0, 0.40)
plt.xlabel('Position (cm)')
# plt.title('profile as a function of time')

line_ani = animation.FuncAnimation(fig, func, init_func = init, interval = 17, frames = 201, blit = True)
# line_ani.save('animation.mp4', writer = 'ImageMagickFileWriter', 
# fps = 30, extra_args=['-vcodec', 'libx264'], dpi = 73, bitrate = 2000)
plt.show()


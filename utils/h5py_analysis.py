import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from multiprocessing import Pool
from scipy.signal import argrelextrema
import scipy.integrate as integrate
import scipy

import h5Parse as h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
BASIC DIGITAL SIGNAL PROCESSING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def find_nearest(array, value):
    """
    PURPOSE
    -------
        This function finds the nearest element to the inputted value in the array.

    INPUTS
    ------
        array: 1d array
            The array in which the desired element will be found.
    
        value: float
            The float which to compare the elements of the array.

    OUTPUTS
    -------
        array[idx]: float
            The value in the array which is closest in value to the 
            inputted value.

        idx: integer
            The index of the element.
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def local_maxima(array):
    """
    PURPOSE
    -------
        This function determines the highest value local maxima of an inputted array 
        along with the index where it is located.

    INPUTS
    -----
        array: 1d array
            The array in which to find the local maxima.
    
    OUTPUTS
    -------
        first_max_idx: integer
            The integer index of where the highest local
            maxima is located.

        second_max_idx: integer
            The integer index of where the second highest local
            maxima is located.

        third_max_idx: integer
            The integer index of where the third highest local
            maxima is located.
    """
    maxInd = argrelextrema(array, np.greater)
    vals = array[maxInd]

    sorted_array = np.sort(vals)
    second_max = sorted_array[-2]
    idx1 = np.argwhere(vals == second_max)
    second_max_idx = maxInd[0][idx1][0][0]

    third_max = sorted_array[-3]
    idx2 = np.argwhere(vals == third_max)
    third_max_idx = maxInd[0][idx2][0][0]

    first_max_idx = np.argmax(array)

    return first_max_idx, second_max_idx, third_max_idx

def highpass_filter(signal, cutoff_freq=10000, sampling_rate=6.25e6):
    """
    PURPOSE
    -------
        This function is a high pass filter for a time signal.

    INPUTS
    ------
        signal: 1d array
            1d time array.

        cutoff_freq: integer or float (optional)
            The frequency cutoff, anything below this will be filtered out.

        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.

    OUTPUTS
    -------
        filtered_signal: 1d array
            1d array of the high pass filtered time signal.
    """
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data)
        return y

    filtered_signal = butter_highpass_filter(signal, cutoff_freq, sampling_rate)
    return filtered_signal

def calculate_fft(signal):
    """
    PURPOSE
    -------
        This function takes a signal and calculates its power spectrum from the FFT.
    
    INPUTS
    ------
        signal: 1d array
            1d time array of desired data to be FFTed. Shape: (6144,)

    OUTPUTS
    -------
        ps: 1d array
            1d array of the power spectrum of the inputted time
            signal in the frequency domain. Shape: (6144,)
    """
    ps = np.abs(np.fft.fft(signal))**2
    return ps
    
def fft_probe(data, sampling_rate=6.25e6):
    """
    PURPOSE
    -------
        This function calculates the Fast Fourier Transform (FFT) of each "pixel" in the 
        Nx x Ny grid along with the shot number, giving Nx*Ny*num_shots FFT arrays which 
        are then averaged in order to plot an averaged FFT the entire Nd array.
    
    INPUTS
    ------
        data: Nd array
            Array with the magnetic field data in the shape of (Ny, Nx, num_shots, time).
            
        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.

    OUTPUTS
    -------
        Outputs a plot of the averaged FFT, (averaged power spectrum vs. frequency).
    """
    ps_list = []
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                signal = data[j, i, k, :]
                ps = calculate_fft(signal)
                ps_list.append(ps)

    avg_ps = np.mean(ps_list, axis=0)         
    time_step = 1.0/sampling_rate
    freqs = np.fft.fftfreq(avg_ps.size, time_step)
    return freqs, avg_ps

def create_syn_data():
    """
    PURPOSE
    -------
        This function creates two synthetic waves in order to test the correlation 
        function's accuracy of finding phase velocity.

    INPUTS
    ------
        None

    OUTPUTS
    -------
        delta_t: integer
            The integer of the difference in time delay from the maximum peak of the 
            correlation wave signal and the center of the correlation function.

        correlation: 1d array
            The 1d array of the correlated wave signals.

        signal1: 1d array
            The 1d array of the first synthetic signal.

        signal2: 1d array
            The 1d array of the second synthetic signal.
    """
    t = np.linspace(0, 500, 500)
    signal1 = np.exp(-1*((t-250)/500)**2)*np.cos(-(2*np.pi*t)/100)
    signal2 = np.exp(-1*((t-(250+2/2))/500)**2)*np.cos(2-(2*np.pi*t)/100)
    correlation = np.correlate(signal1, signal2, 'full')
    max_idx = np.argmax(correlation)

    delta_t = (len(correlation)/2 + 1) - max_idx

    return delta_t, correlation, signal1, signal2

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
OPEN HDF5 DATA: B PROBE AND LANGMUIR PROBE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def bit_to_voltage(signal):
    """
    PURPOSE
    -------
        This functions converts the signal from bits to voltage then normalizes by
        subtracting the mean value.
    
    INPUTS
    ------
        signal: Nd array
            The desired signal to convert
    
    OUTPUTS
    -------
        signal: Nd array
            The normalized converted array in volts.
    """
    signal = (signal - 2.0**15)*(5.0/2.0**16)
    signal = signal - np.mean(signal)
    return signal

def reshape_data(f, num_shots):
    """
    PURPOSE
    -------
        The simplest function to reshape data into appropriate form based on the
        number of shots.

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the data is located
            
        num_shots: integer (optional)
            Number of shots 
    
    OUTPUTS
    -------
        data: 3d array
            Reshaped data in the form of (position, shot num, time signal)
    """
    data = h.openHDF5_dataset(f)['data']
    data = [data[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(data.shape[0])/num_shots))]
    data = bit_to_voltage(np.array(data))
    return data

def get_motion_list(f, choose=True, index=0):
    """
    PURPOSE
    -------
        This function pulls out the motion lists of the user 
        specified probe and returns a dictionary with all the 
        attributes and data.

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located

        choose: Boolean (optional)
            If True, then the motion list obtained is user specified. If False, then
            the motion list choosen is by the argument "index."

        index: integer (optional)
            If choose == False, then the motion list obtained is by the inputted index.

    OUTPUTS
    -------
        d: dictionary
            A dictionary of the specified motion list.
    """
    dsKeys = f['Raw data + config']['6K Compumotor'].keys()
    dsets = f['Raw data + config']['6K Compumotor']
    if choose == True:
        for it in range(len(dsKeys)):
            print '{}  : {}'.format(it, dsKeys[it])
            print ".........................."
        choice = str(input("Choose a dataset: \n"))
    else:
        choice = str(index)     
    if choice.isdigit():
        if int(choice) in range(len(dsKeys)):
            if int(choice) < 3:
                dset = dsets[dsKeys[int(choice)]]
                d = {}
                attrs = dset.attrs.values()
                d['Motion list'] = attrs[0]
                d['Motion count'] = attrs[1]
                d['Data motion count'] = attrs[2]
                d['Created date'] = attrs[3]
                d['Grid center x'] = attrs[4]
                d['Grid center y'] = attrs[5]
                d['Delta x'] = attrs[6]
                d['Delta y'] = attrs[7]
                d['Nx'] = attrs[8]
                d['Ny'] = attrs[9]
                return d
            else:
                dset = dsets[dsKeys[int(choice)]]
                d = {}
                d['Shot number'] = dset['Shot number']
                d['x'] = dset['x']
                d['y'] = dset['y']
                d['z'] = dset['z']
                d['theta'] = dset['theta']
                d['phi'] = dset['phi']
                d['Motion list'] = dset['Motion list']
                d['Probe name'] = dset['Probe name']
                return d
        else:
            print "Dataset not found..."
            return -1
    else:
        print "Not a number."
        return -1

def data_parser_B_field(f, num_shots=15, gain=10):
    """
    PURPOSE
    -------
        This function splits the data into an array of dimensions (Ny, Nx, num_shots, time)
        where each of the different probe's initial locations are taken to be the 
        upper most right corner of the Nx x Ny grid and therefore the shots are not
        in order.

    INPUTS
    ------
        file: HDF5 file
            The desired HDF5 file with the data.

        num_shots: integer (optional)
            Number of shots taken at each position

        gain: integer or float (optional)
            The gain on the magnetic probe voltage.
    
    OUTPUTS
    -------
        reshaped_data: Nd array
            Array of the same data reshaped to be spatially valid in the form of
            (Ny, Nx, num_shots, time)
    """
    data_struct = h.openHDF5_dataset(f)
    data = data_struct['data']
    dataType = data_struct['data type']

    d = get_motion_list(f, choose=False, index=0)
    Nx = d['Nx']
    Ny = 3*d['Ny']
    
    if dataType[-1] == '4' or \
       dataType[-1] == '3':
        print "PROBE "+dataType[-1]+", CONFIGURATION "+dataType[0:2]
        shot_list = [data[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(data.shape[0])
            /num_shots))]
        reshaped_data = np.zeros((Ny, Nx, num_shots, data.shape[1]))
        iteration = 0
        for i in range(0, Nx):
            for j in range(0, Ny):
                reshaped_data[j, i, :, :] = shot_list[iteration]
                iteration += 1
        reshaped_data = gain*bit_to_voltage(reshaped_data)
        return reshaped_data

    if dataType[-1] == '5':
        print "PROBE "+dataType[-1]+", CONFIGURATION "+dataType[0:2]
        shot_list = [data[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(data.shape[0])
            /num_shots))]
        shot_split1 = shot_list[0:1350]
        shot_split2 = shot_list[1350:]
        for i in range(0, len(shot_split1)):
            shot_split2.append(shot_split1[i])
        reshaped_data = np.zeros((Ny, Nx, num_shots, data.shape[1]))
        iteration = 0
        for i in range(0, Nx):
            for j in range(0, Ny):
                reshaped_data[j, i, :, :] = shot_split2[iteration]
                iteration += 1
        reshaped_data = gain*bit_to_voltage(reshaped_data)
        return reshaped_data

    if dataType[-1] == '8':
        print "PROBE "+dataType[-1]+", CONFIGURATION "+dataType[0:2]
        shot_list = [data[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(data.shape[0])
            /num_shots))]
        shot_split1 = shot_list[0:675]
        shot_split2 = shot_list[675:]
        for i in range(0, len(shot_split1)):
            shot_split2.append(shot_split1[i])
        reshaped_data = np.zeros((Ny, Nx, num_shots, data.shape[1]))
        iteration = 0
        for i in range(0, Nx):
            for j in range(0, Ny):
                reshaped_data[j, i, :, :] = shot_split2[iteration]
                iteration += 1
        reshaped_data = gain*bit_to_voltage(reshaped_data)
        return reshaped_data

    else:
        print "Dataset not found."

def data_parser_langmuir(f, num_shots=15, gain=50, resistance=110):
    """
    PURPOSE
    -------
        This function opens the Langumir probe HDF5 data and reshapes the data based
        on the number of shots. This function additionally converts the bits into voltage
        and converts into orignal signal by multiplying voltage by gain and dividing the
        resistance by the current in order to get the current.
    
    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located

        num_shots: integer (optional)
            Number of shots taken at each position.
    
       gain: integer or float (optional)
            The gain on voltage of the Langmuir probe.

       resistance: integer or float (optional)
            The resistance used for the current on the Langmuir probe.

    OUTPUTS
    -------
        voltage_array: 3d array
            Voltage array in volts formatted as: (position, shot num, time signal)
        
        current_array: 3d array
            Current array in Amperes formatted as: (position, shot num, time signal)
    """
    print "..................................."
    print "CHOOSE VOLTAGE ARRAY"
    print "..................................."
    voltage = h.openHDF5_dataset(f)['data']
    print "..................................."
    print "CHOOSE CURRENT ARRAY"
    print "..................................."
    current = h.openHDF5_dataset(f)['data']

    voltage_list = [voltage[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(voltage.shape[0])
        /num_shots))]
    current_list = [current[num_shots*i:num_shots*i+num_shots] for i in range(0, int(math.ceil(current.shape[0])
        /num_shots))]
    voltage_array = np.array(voltage_list)
    current_array = np.array(current_list)

    voltage_array = gain*bit_to_voltage(voltage_array)
    current_array = -1*gain*(bit_to_voltage(current_array)/resistance)

    return voltage_array, current_array

def generalized_data(f, num_shots=15, gain=10):
    """
    PURPOSE
    -------
       This function is the most general format way of pulling out
       the desired data out of the HDF5 file. Relies on knowning the
       motion list of the probe.
       NOTE: This does not account for the voltage amplification, need
       to multiply the signal by the gain!

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located
            
        num_shots: integer
            Number of shots 

        gain: integer or float (optional)
            The gain on the probe voltage.

    OUTPUTS
    -------
       data_points: list of lists
            A list of list in the format: [[x, y, shot number, time signal], ...]
    """
    data_struct = h.openHDF5_dataset(f)
    data = data_struct['data']
    dataType = data_struct['data type']
    if dataType[-1] == '4' or \
       dataType[-1] == '3':
        motion_list = get_motion_list(f, choose=False, index=9)
    if dataType[-1] == '5':
        motion_list = get_motion_list(f, choose=False, index=8)
    if dataType[-1] == '8':
        motion_list = get_motion_list(f, choose=False, index=7)

    if dataType[-3] == 'Vsweep' or \
       dataType[-3] == 'Isat' or \
       dataType[-3] == 'Isweep':
        motion_list = gte_motion_list(f, choose=False, index=8)
        
    x_motion = motion_list['x']
    y_motion = motion_list['y']

    data_points = []
    shot = 1
    for i in range(0, data.shape[0]):
        signal = gain*bit_to_voltage(data[i])
        data_points.append([x_motion[i], y_motion[i], shot, signal])
        if i % num_shots == 0:
            shot = 1
        shot += 1
                           
    return data_points

def generalized_reshaped_data(f, num_shots, Nx=45, Ny=45):
    """
    PURPOSE
    -------
        This function takes a HDF5 file and produces a Nx by Ny grid with each corresponding
        shot and time signal and creates a 4d array of it.

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located
            
        num_shots: integer
            Number of shots 

        Nx: float (optional)
            Number of x points

        Ny: float (optional)
            Number of y points

    OUTPUTS
    -------
        array: 4d array
            4d array of (x, y, num_shots, time signal)
    """
    data = generalized_data(f, num_shots=num_shots)
    row_data = [data[num_shots*i:num_shots*i+num_shots] for i in range(0, len(data)/num_shots)]
    split_data = [row_data[Nx*i:Nx*i+Nx] for i in range(0, len(row_data)/Nx)]
    array = np.zeros((Ny, Nx, num_shots, len(data[0][3])))
    for i in range(0, Nx):
        for j in range(0, Ny):
            for k in range(0, num_shots):
                array[j, i, k, :] = split_data[j][i][k][3]

    return array

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CONVERTING TO B FIELD UNITS: B PROBE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def calc_B_ifft(data, sampling_rate=6.25e6, cutoff_freq=10000):
    """
    PURPOSE
    -------
        This function calculates the magnetic field from the Bdot using inverse Fourier 
        Transform method. This function also filters out frequencies below 10 KHz.

    INPUTS
    ------
        data: Nd array
            Bdot data from a probe
        
        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.
        
        cutoff_freq: integer or float (optional)
            The frequency cutoff, anything below this will be filtered out.

    OUTPUTS
    -------
        inverse_fft: Nd array
           The B (in one direction) for that probe.
    """
    time_step = 1.0/sampling_rate

    inverse_fft = np.zeros((data.shape[1], data.shape[0], data.shape[3]))
    fft_list = []
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            fft_list = []
            for k in range(0, data.shape[2]):
                signal = data[j, i, k, :]
                fft = np.fft.fft(signal)
                fft_list.append(fft)

            avg_fft = np.mean(fft_list, axis=0)
            freqs = np.fft.fftfreq(avg_fft.size, time_step)
            vals = np.argwhere(freqs < cutoff_freq)

            for m in range(0, len(vals)):
                idx = vals[m]
                avg_fft[idx] = 0
        
            np.seterr(all='ignore')
            
            B_fft = avg_fft/(-1j*2*np.pi*freqs)
            B_fft[0] = 0+0j
            inverse_fft[j, i] = -1*np.fft.ifft(B_fft)
            
    inverse_fft = inverse_fft*1e5
    return inverse_fft

def calc_B_integration(data, sampling_rate=6.26e6, nA=1e-6, animation=False):
    """
    PURPOSE
    -------
        This function calculates the magnetic field B using the method of integration 
        of the Bdot signal, then passes it through a high pass filter.

    INPUTS
    ------
        data: Nd array
           Bdot data from probe.
        
        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.
            
        nA: float (optional)
        	The number of turn, n, multipled by the probe's area, A, (should be in m^2).

        animation: Boolean (optional)
           If true, then function animates the B field through time.

    OUTPUTS
    -------
        filtered_signal: Nd array
            high pass filtered B field of probe.
    """
    B_fields = np.zeros((data.shape[0], data.shape[1], data.shape[3]))
    delta_t = 1.0/sampling_rate
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            shot_list = []
            for k in range(0, data.shape[2]):
                signal = data[j, i, k, :]
                shot_list.append(signal)
            
            avg_data = np.mean(shot_list, axis=0)
            t = delta_t*np.arange(data.shape[3])
            B_int = integrate.cumtrapz(avg_data, t, initial=0)
            B_fields[j, i] = (B_int/nA)

    filtered_signal = highpass_filter(B_fields)
    if animation == True:
        animate_time_signal(filtered_signal)
    
    return filtered_signal

def calculate_mag(f, ifft=False):
    """
    PURPOSE
    -------
        This function calculates the magnitude of the magnetic field by 
        either using the inverse FFT or integration method.

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located

        ifft: Boolean (optional)
            If True, then function will calculate the magnetic field
            using the inverse FFT method. If False, the function will 
            calculate the magnetic field using the integration method.

    OUTPUTS
    -------
        mag: 3d array
            The 3d array of the magnitude of the B field calculated by:
            (Bx^2 + By^2)^(1/2). Array in format: (y, x, time)
    """
    print "CHOOSE X COMPONENT OF B FIELD: "
    print "..................................."
    datax = data_parser_B_field(f)
    print "CHOOSE Y COMPONENT OF B FIELD: "
    print "..................................."
    datay = data_parser_B_field(f)
    if ifft == True:
        x = calc_B_ifft(datax)
        y = calc_B_ifft(datay)
    else:
        x = calc_B_integration(datax)
        y = calc_B_integration(datay)
    mag = np.sqrt(x**2 + y**2)
    return mag

def vector_field(data_x, data_y, plot=False, index=2000):
    """
    PURPOSE
    -------
        This function calculates the x and y components of the vector field of the B 
        field component as well as the magntitude of the total B field.

    INPUTS
    ------
        data_x: Nd array
            Magnetic probe in the x direction's data.

        data_y: Nd array
            Magnetic probe in the y direction's data.

        plot: Boolean (optional)
            If true, then function plots the magnitude and overlaps the vector field 
            at specified point in time (index).

        index: integer (optional)
            If plotted, this will be the position in time.

    OUTPUTS
    -------
        magnitude: 3d array
            Magnitude of B field in the form: (x position, y position, time)

        u: 3d array
            x component of the magnetic field: (x position, y position, time)

        v: 3d array 
            y component of the magnetic field: (x position, y position, time)
    """
    ifft_x = calc_B_ifft(data_x)
    ifft_y = calc_B_ifft(data_y)

    magnitude = np.zeros((ifft_x.shape[1], ifft_x.shape[0], ifft_x.shape[2]))
    u = np.zeros((ifft_x.shape[1], ifft_x.shape[0], ifft_x.shape[2]))
    v = np.zeros((ifft_x.shape[1], ifft_x.shape[0], ifft_x.shape[2]))
    for i in range(0, ifft_x.shape[0]):
        for j in range(0, ifft_x.shape[1]):
            u[j, i] = ifft_x[j, i]
            v[j, i] = ifft_y[j, i]
            magnitude[j, i] = np.sqrt(ifft_x[j, i]**2 + ifft_y[j, i]**2)
            
    x = np.arange(ifft_x.shape[0])
    y = np.arange(ifft_x.shape[1])
    if plot == True:
        plt.imshow(magnitude[:, :, index], interpolation='catrom')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('|B| for one moment in time with vector field')
        plt.quiver(x, y, v[:, :, index], u[:, :, index])
        plt.show()

    return magnitude, u, v

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FINDING PHASE DIFFERENCE BETWEEN B PROBES
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def correlation_function(f, pos, idx, sampling_rate=6.26e6, delta_x=2.875):
    """
    PURPOSE
    -------
        This function calculates the correlation function of the two inputted probes.

    INPUTS
    ------
        f: HDF5 file
            The desired HDF5 file with the data.

        pos: 2 element list
            List of the x and y desired position -> [x, y].
    
        idx: 2 element list
            List of the first and last index of the time signal. The purpose behind this 
            is to single out the high frequency wave and the low frequency wave.

        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.
            
        delta_x: float (optional)
        	Distance between the two probes (in meters).
            
    OUTPUT
    ------
        v: float
            The phase velocity of the correlated signal.

        phase: float
            The phase between the two signals.

        k_parallel: float
            The corresponding k parallel value of the correlated signal.

        omega: float
            Angular frequency of the correlated signal.
    """
    x, y = pos[0], pos[1]
    first_idx, last_idx = idx[0], idx[1]
    
    print ".........................."
    print "PICK FIRST PROBE: "
    print ".........................."
    probe1 = data_parser_B_field(f)
    print ".........................."
    print "PICK SECOND PROBE: "
    print ".........................."
    probe2 = data_parser_B_field(f)

    correlation_list = []
    for i in range(0, probe1.shape[2]):
        signal1 = probe1[y, x, i, first_idx:last_idx]
        signal2 = probe2[y, x, i, first_idx:last_idx]
        correlation_list.append(np.correlate(signal1, signal2,'full'))
        
    correlation = np.mean(correlation_list, axis=0)
    max_idx = np.argmax(correlation)

    t = (len(correlation)/2) - max_idx
    delta_t = 1.0/sampling_rate
    t = t*delta_t
    v = delta_x/t

    ps = calculate_fft(correlation, sampling_rate=sampling_rate)
    max_freq_idx = np.argmax(ps)
    freqs = np.fft.fftfreq(len(correlation), 1.0/sampling_rate)
    frequency = freqs[max_freq_idx]
    
    phase = (2*np.pi*frequency*delta_x)/v
    k_parallel = phase/delta_x
    omega = 2*np.pi*frequency
    
    return v, phase, k_parallel, omega

def spectral_density_function(f, pos, idx, sampling_rate=6.26e6):
    """
    PURPOSE
    -------
        This function calculates the spectral density function of two different probes 
        (of the same magnetic field component) at a specified position on the Nx x Ny 
        grid at the desired time of the signal. This is done by finding the FFT of each 
        position and shot of each probe then multiplying one probe's FFT by the other 
        probe's FFT then averaging. 

    INPUTS
    ------
        f: HDF5 file
            The desired HDF5 file with the data.

        pos: 2 element list
            List of the x and y desired position -> [x, y].
    
        idx: 2 element list
            List of the first and last index of the time signal. The purpose behind this 
            is to single out the high frequency wave and the low frequency wave.
            
        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.

    OUTPUTS
    -------
        phi: 1d array
            1d array of the phi values as a function of angular frequency. 

        gamma: 1d array
            1d array of the gamma values, the ratio of the 
            chi^2 / (fft of 1st probe^2 * fft of 2nd probe^2). The highest it can 
            possibly be is 1.

        chi: 1d array
            1d array of the chi values, the averaged multiplied FFTs of the two probes.
    
        freqs: 1d array
            1d array of all the frequency values found from the FFT.
    """
    x, y = pos[0], pos[1]
    first_idx, last_idx = idx[0], idx[1]
    print ".........................."
    print "PICK FIRST PROBE: "
    print ".........................."
    probe1 = data_parser_B_field(f)
    print ".........................."
    print "PICK SECOND PROBE: "
    print ".........................."
    probe2 = data_parser_B_field(f)

    time_step = 1.0/sampling_rate
    freqs = np.fft.fftfreq(probe1.shape[3], time_step)

    ps1_list_sqr = []
    ps2_list_sqr = []
    phi_phibar_list = []
    for i in range(0, probe1.shape[2]):
        signal1 = probe1[y, x, i, first_idx:last_idx]
        signal2 = probe2[y, x, i, first_idx:last_idx]
        fft1 = np.fft.fft(signal1)
        fft2 = np.fft.fft(signal2)
        ps1_list_sqr.append(np.abs(fft1)**2)
        ps2_list_sqr.append(np.abs(fft2)**2)
        phi_phibar_list.append(fft1*np.conj(fft2))

    chi = np.mean(phi_phibar_list, axis=0)
    phi = np.arctan(chi.imag/chi.real)
    
    ensemble_avg_1 = np.mean(ps1_list_sqr, axis=0)
    ensemble_avg_2 = np.mean(ps2_list_sqr, axis=0)
    gamma = np.abs(chi)**2/(ensemble_avg_1*ensemble_avg_2)

    return phi, gamma, chi, freqs

def sdf_analysis(f, pos, delta_x=2.875):
    """
    PURPOSE
    -------
        This function calculates the spectral density function of the two probes at both
        the higher and lower frequency wave and finds the 3 highest peaks in the higher 
        frequency wave and the highest peak in the lower frequency wave (user specified). 
        It calculates the frequency, phase, k parallel, and phase velocity at those points. 
        The dispersion relation (w/k parallel = v) is plotted with a best fit line.

    INPUTS
    ------
        f: HDF5 file
            The desired HDF5 file with the data.

        pos: 2 element list
            List of the x and y desired position -> [x, y].
             
        delta_x: float (optional)
        	Distance between the two probes (in meters).

    OUTPUTS
    -------
        Outputs a plot of the dispersion relation angular frequency vs. k parallel
        with the slope defined as the Alfven speed.
    """
    # higher frequency wave occurs at indices 650 to 900
    phi_high, gamma_high, chi_high, freqs = spectral_density_function(f, pos, [650,900])
    # lower frequency wave occurs at indices 1370 to 2252
    phi_low, gamma_low, chi_low, freqs = spectral_density_function(f, pos, [1370,2252])

    ps_chi_high= np.log10(np.abs(chi_high)**2)
    ps_chi_low = np.log10(np.abs(chi_low)**2)

    # indices found visually
    pos1, idx1 = ps_chi_high[10], 10
    pos2, idx2 = ps_chi_high[13], 13
    pos3, idx3 = ps_chi_low[8], 8

    f1 = freqs[idx1]
    f2 = freqs[idx2]
    f3 = freqs[idx3]

    p1 = phi_high[idx1]
    p2 = phi_high[idx2]
    p3 = phi_low[idx3]

    k_par1 = p1/delta_x
    k_par2 = p2/delta_x
    k_par3 = p3/delta_x

    v_par1 = (2*np.pi*f1*delta_x)/p1
    v_par2 = (2*np.pi*f2*delta_x)/p2
    v_par3 = (2*np.pi*f3*delta_x)/p3

    print "Log(abs(chi)**2)|  frequency (Hz) |  phase (rad) |  k par (m^-1) |  Phase V (m/s) |"
    print "-----------------------------------------------------------------------------------"
    print pos1, f1, p1, k_par1, v_par1
    print pos2, f2, p2, k_par2, v_par2
    print pos3, f3, p3, k_par3, v_par3

    frequencies = [2*np.pi*f1, 2*np.pi*f2, 2*np.pi*f3]
    k_parallels = [-1*k_par1, -1* k_par2, -1*k_par3]

    plt.scatter(k_parallels, frequencies)
    plt.plot(np.unique(k_parallels), np.poly1d(np.polyfit(k_parallels, frequencies, 1))(np.unique(k_parallels)),'--')
    m, b = np.polyfit(k_parallels, frequencies, 1)
    print "Alfven velocity: "+str(m)+" m/s"
    plt.xlabel('K Parallel (m$^{-1}$)')
    plt.ylabel(r'$\omega$ (rad/s)')
    plt.title('Dispersion Relation: $\omega$ = k$_{\|\|}$V$_{A}$')
    plt.tight_layout()
    plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FINDING TEMPERATURE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def temperature_curve_fit(voltage, current, vmax=20):
    """
    PURPOSE
    -------
        This function goes hand in hand with the function calculate_temperature_profile. 
        This function curve fits the portion of voltage and current plot to an exponential
        and returns the coeffients of the fitted curve.

    INPUTS
    ------
        voltage: 1d array
            A 1d array of 1 pulsed signal of the voltage.

        current: 1d array
            A 1d array of the same 1 pulsed signal of the current.

        vmax: integer or float
            A value set to be the upper cutoff for the curve.

    OUTPUTS
    -------
        a[0]: float
            Coeffient A from the fitted curve: Current = Aexp(B*Voltage).

        a[1]: float
            Coeffient b from the fitted curve: Current = Aexp(B*Voltage)
    """
    vals = np.argwhere(voltage < vmax)
    offset = np.abs(np.min(current))
    cropped_voltage = []
    cropped_current = []
    for i in range(0, len(vals)):
        idx = vals[i][0]
        cropped_voltage.append(voltage[idx])
        cropped_current.append((current[idx] + offset))

    a, b = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t), cropped_voltage, cropped_current, p0=(3, 1e-4))

    fitted_data = []
    for i in range(0, len(cropped_current)):
        fitted_data.append(a[0]*np.exp(a[1]*cropped_voltage[i]))

    return a[0], a[1]

def calculate_temperature_profile(voltage, current, plot=True, vmax=20, xmin=-30, xmax=40.5, delta_x=0.5):
    """
    PURPOSE
    -------
        This function averages the 15 shots for both the current and voltage sweep data 
        then curve fits an exponential to the V vs. I plot to calculate the temperature 
        profile of the plasma.

    INPUTS
    ------
        file: HDF5 file
            The desired HDF5 file with the data.

        plot: Boolean
            If True then function plots the temperature vs. position. 

        vmax: integer or float
            A value set to be the upper cutoff for the curve.

        xmin: float (optional)
            The minimum position in cm of the x axis.

        xmax: float (optional)
            The maximum position in cm of the x axis.

        delta_x: float (optional)
            The change in the x positions in cm.

    OUTPUTS
    -------
        A_list: list
            List of the coeffienct A of the fitted curve. I = Aexp(B*V)
    
        avg_temp_list: list
            List of the averaged temperatures of the three pulses of the fitted curve.
    """
    voltages = np.mean(voltage, axis=1)
    currents = np.mean(current, axis=1)

    A_list = []
    temp_list_1 = []
    temp_list_2 = []
    temp_list_3 = []
    for i in range(0, voltages.shape[0]):
        voltage1, current1 = voltages[i][40000:43116], currents[i][40000:43116]
        a1, b1 = temperature_curve_fit(voltage1, current1, vmax=vmax)
        A_list.append(a1)
        temp_list_1.append(1/b1)
        voltage2, current2 = voltages[i][43303:46246], currents[i][43303:46246]
        a2, b2 = temperature_curve_fit(voltage2, current2, vmax=vmax)
        A_list.append(a2)
        temp_list_2.append(1/b2)
        voltage3, current3 = voltages[i][46339:49363], currents[i][46339:49363]
        a3, b3 = temperature_curve_fit(voltage3, current3, vmax=vmax)
        A_list.append(a3)
        temp_list_3.append(1/b3)

    total_temp_list = [temp_list_1, temp_list_2, temp_list_3]
    avg_temp_list = np.mean(total_temp_list, axis=0)

    if plot == True:
        position = np.arange(xmin, xmax, delta_x)
        plt.plot(position, avg_temp_list)
        plt.title('Averaged Temperature vs. Position')
        plt.ylabel('Temperature (eV)')
        plt.xlabel('Position (cm)')
        plt.show()

    return avg_temp_list

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PLOTTING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def basic_2d_plot(data, time_idx, individual_shot=False, shot_num = 0, contour=True, xmin=-19.8, 
    xmax=20.7, ymin=-19.8, ymax=20.7, delta_x=0.9, delta_y=0.9):
    """
    PURPOSE
    -------
        This function serves as a very basic 2d plotting, meant for 3d or 4d arrays.
        ex. (x, y, time) or (x, y, shot number, time).

    INPUTS
    ------
        data: Nd array
            Data from a probe

        time_idx: integer
            The index of the time array desired. (Plots only 1 instance in time)

        individual_shot: Boolean (optional)
            False = data has been averaged over all shots.
            True = pick a shot number to animate that data.
    
        shot_num: integer (optional)
            If individual_shot = True, then specify which shot to animate. Default is 0.

        contour: Boolean (optional)
            If True, then function overlays the contour plot.
    
        xmin: float (optional)
            The minimum position in cm of the x axis.

        xmax: float (optional)
            The maximum position in cm of the x axis.

        ymin: float (optional)
            The minimum position in cm of the y axis.

        ymax: float (optional)
            The maximum position in cm of the y axis.

        delta_x: float (optional)
            The change in the x positions in cm.

        delta_y: float (optional)
            The change in the y positions in cm.

    OUTPUTS
    -------
        2d plot of the data at one instance in time
    """
    if individual_shot == False:
        plt.imshow(data[:, :, time_idx], extent=[xmin, xmax, ymin, ymax], interpolation='catrom')
        if contour == True:
            x = np.arange(xmin, xmax, delta_x)
            y = np.arange(xmin, xmas, delta_y)
            plt.contour(x, y, data[:, :, time_idx])
        plt.colorbar()
        plt.clim(min(data[22, 22, :]), max(data[22, 22, :]))
    else:
        plt.imshow(data[:, :, shot_num, time_idx], extent=[xmin, xmax, ymin, ymax], interpolation='catrom')
        plt.clim(min(data[22, 22, shot_num, :]), max(data[22, 22, shot_num, :]))
        plt.colorbar()
        if contour == True:
            x = np.arange(xmin, xmax, delta_x)
            y = np.arange(xmin, xmax, delta_y)
            plt.contour(x, y, data[:, :, shot_num, time_idx])
        
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.show()
    
def animate_time_signal(data, interval=30, individual_shot=False, shot_num=0, xmin=-19.8, xmax=20.7, 
    ymin=-19.8, ymax=20.7, delta_x=0.9, delta_y=0.9, cmap='plasma'):
    """
    PURPOSE
    -------
        This function animates the inputed signal, meant for the B field but can be used
        for any 3d array. Note: this animation is not real time, to make it real time 
        change the "interval" argument in the FuncAnimation. Changing "blit" to True
        will make the animation replay indefinetly. Additionally changing the x and y
        ticks to a desired range can be done by adding "extend=[min x, max x, min y, max y]"
        in the plt.imshow function's arguments.
        
    INPUTS
    ------
        data: 3d or 4d array
            A 3d or 4d array meant to be formatted as: (x, y, time) or (x, y, shot_num, time)

        interval: integer (optional)
            Change this number to speed up or slow down the animation.

        individual_shot: Boolean (optional)
            False = data has been averaged over all shots.
            True = pick a shot number to animate that data.
    
        shot_num: integer (optional)
            If individual_shot = True, then specify which shot to animate. Default is 0.
    
    OUTPUTS
    -------
        An animation of the signal through time.
    """
    fig = plt.figure()
    if individual_shot == False:
        im = plt.imshow(data[:, :, 0], interpolation='catrom', extent=[xmin, xmax, ymin, ymax], cmap=cmap)
        plt.colorbar()
        plt.clim(min(data[22, 22, :]), max(data[22, 22, :])-0.15)
    else:
        im = plt.imshow(data[:, :, shot_num, 0], interpolation='catrom', extent=[xmin, xmax, ymin, ymax], cmap=cmap)
        plt.colorbar()
        plt.clim(min(data[22, 22, shot_num, :]), max(data[22, 22, shot_num, :])-0.15)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')

    def updatefig(j):
        if individual_shot == False:
            im.set_array(data[:, :, j])
        else:
            im.set_array(data[:, :, shot_num, j])
        return im

    ani = animation.FuncAnimation(fig, updatefig, interval=interval, blit=False)
    plt.tight_layout()
    plt.show()

def animation_vector_field(data, u, v, interval=30, individual_shot=False, shot_num=0):
    """
    PURPOSE
    -------
        This function allows an animation through time of the 3d array as well as an 
        animation of the vector fields for the corresponding data.

    INPUTS
    ------
        data: 3d array 
            The desired data to be animated, should be in the shape:
            (x position, y position, time signal).

        u: 3d array
            The x component of the magnetic field (vector field). Should be in the shape 
            of (x position, y position, time signal).

        v: 3d array
            The y component of the magnetic field (vector field). Should be in the shape 
            of (x position, y position, time signal).
            
        interval: integer (optional)
        	Change this number to speed up or slow down the animation.

        individual_shot: Boolean (optional)
            False = data has been averaged over all shots.
            True = pick a shot number to animate that data.
    
        shot_num: integer (optional)
            If individual_shot = True, then specify which shot to animate. Default is 0.

    OUTPUTS
    -------
        Animation through time of data with the vector field.
    """
    fig = plt.figure()
    if individual_shot == False:
        im = plt.imshow(data[:, :, 0], interpolation='catrom')
    else:
        im = plt.imshow(data[:, :, shot_num, 0], interpolation='catrom')
    plt.colorbar()
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    if individual_shot == False:
        im2 = plt.quiver(x, y, u[:, :, 0], v[:, :, 0])
    else:
        im2 = plt.quiver(x, y, u[:, :, shot_num, 0], v[:, :, shot_num, 0])
    plt.xlabel('X')
    plt.ylabel('Y')

    def updatefig(j):
        if individual_shot == False:
            im.set_array(data[:, :, j])
            im2.set_UVC(u[:, :, j], v[:, :, j])
        else:
            im.set_array(data[:, :, shot_num, j])
            im2.set_UVC(u[:, :, shot_num, j], v[:, :, shot_num, j])
            
        return im, im2

    ani = animation.FuncAnimation(fig, updatefig, interval=interval, blit=False)
    plt.tight_layout()
    plt.show()

def plot_fft(data):
    """
    PURPOSE
    -------
        This function takes the outputs of the function "fft_parallel" and plots it as 
        averaged power spectrum vs. frequency.
    
    INPUTS
    ------
        data: Nd array
            Array with the magnetic field data in the shape of (Ny, Nx, num_shots, time).

    OUTPUTS
    -------
        Outputs a plot of the averaged FFT, (averaged power spectrum vs. frequency).
    """
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    frequency, ps = fft_probe(data)
    
    plt.plot(np.log10(frequency), np.log10(ps))
    plt.title('FFT of Averaged Pixels')
    plt.xlabel('log10(frequency)')
    plt.ylabel('log10(power)')
    plt.show()
    
def plot_freq_spatially(data, desired_freq, sampling_rate=6.25e6):
    """
    PURPOSE
    -------
        This function plots the inputted frequency spatially on the Nx x Ny grid in order 
        to see the intensity of the signal at that frequency.

    INTPUTS
    ------
        data: Nd array
            Array with the magnetic field data in the shape of (Ny, Nx, num_shots, time).

        desired_freq: float
            The desired frequency to look at in Hz.
            
        sampling_rate: integer (optional)
            The rate at which sampling was done of the signal.

    OUTPUTS
    -------
         Outputs a 2d plot of the intensity of the signal at the inputted frequency on the 
         Nx x Ny grid.
    """
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    spatial_ps = np.zeros((data.shape[0], data.shape[1]))
    time_step = 1.0/sampling_rate
    freqs = np.fft.fftfreq(data.shape[3], time_step)
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            ps_temp_list = []
            for k in range(0, data.shape[2]):
                signal = data[j, i, k, :]
                ps = np.abs(np.fft.fft(signal))**2
                ps_temp_list.append(ps)
            avg_ps = np.mean(ps_temp_list, axis=0)
            idx = np.argwhere(freqs > desired_freq)
            index = idx[0][0]
            spatial_ps[j, i] = ps[index]
    plt.imshow(spatial_ps)
    plt.title('Spatial Plot of Frequency '+str(desired_freq))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
def plot_sdf(f, pos):
    """
    PURPOSE
    -------
        This function takes the outputs of the spectral density function of both the high 
        and low frequency wave and plots the log10(chi**2) vs. log10(frequency), 
        gamma vs. log10(frequency), and cos(phi) vs. log10(frequency) giving 6 plots.

    INPUTS
    ------
        f: HDF5 file
            The desired HDF5 file with the data.

        pos: 2 element list
            List of the x and y desired position -> [x, y].

    OUTPUTS
    -------
        This functions outputs 6 plots, 3 for each frequency wave of the different 
        properties as a function of the log10(frequency).
    """
    # higher frequency wave occurs at indices 650 to 900
    phi_high, gamma_high, chi_high, freqs = spectral_density_function(f, pos, [650,900])
    # lower frequency wave occurs at indices 1370 to 2252
    phi_low, gamma_low, chi_low, freqs = spectral_density_function(f, pos, [1370,2252])
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    log_freq = np.log10(2*np.pi*freqs)
    
    fig = plt.figure()
    plt.subplot(321)
    plt.plot(log_freq, np.log10(np.abs(chi_high)**2))
    plt.ylabel('Log(|$\chi$|$^{2}$)')
    plt.xlabel('Log($\omega$)')
    plt.title('Higher Frequency Wave')
    
    plt.subplot(323)
    plt.plot(log_freq, gamma_high)
    plt.xlabel('Log($\omega$)')
    plt.ylabel('$\gamma$')
    
    plt.subplot(325)
    plt.plot(log_freq, np.cos(phi_high))
    plt.xlabel('Log($\omega$)')
    plt.ylabel('Cos($\phi$)')

    plt.subplot(322)
    plt.plot(log_freq, np.log10( np.abs(chi_low)**2))
    plt.ylabel('Log(|$\chi$|$^{2}$)')
    plt.xlabel('Log($\omega$)')
    plt.title('Lower Frequency Wave')
    
    plt.subplot(324)
    plt.plot(log_freq, gamma_low)
    plt.xlabel('Log($\omega$)')
    plt.ylabel('$\gamma$')
    
    plt.subplot(326)
    plt.plot(log_freq, np.cos(phi_low))
    plt.xlabel('Log($\omega$)')
    plt.ylabel('Cos($\phi$)')
    
    plt.tight_layout()
    plt.show()
        
def plot_current_voltage(voltage, current, idx, vmax=20, cutoff_min=46339, cutoff_max=49363):
    """
    PURPOSE
    -------
        This function takes the voltage and current of the data run and curve fits an 
        exponential and plots them.

    INPUTS
    -----
        file: HDF5 file
            The desired HDF5 file with the data.

        idx: integer
            The index of the position desired for the current and voltage
            
        cutoff_min: integer
        	Index to cutoff the voltage array. This is meant as a way of isolating a 
        	single pulse. (In the dataset used, there are three).
        	
        cutoff_max: integer
        	Index to cutoff the voltage array. Same reasoning as in cutoff_min description.

    OUTPUTS
    ------
        Plot of cropped voltage vs. current along with the exponential curve fit.
    """
    voltages = np.mean(voltage, axis=1)
    currents = np.mean(current, axis=1)

    voltage, current = voltages[idx][cutoff_min:cutoff_max], currents[idx][cutoff_min:cutoff_max]

    vals = np.argwhere(voltage < vmax)
    offset = np.abs(np.min(current))
    cropped_voltage = []
    cropped_current = []
    for i in range(0, len(vals)):
        idx = vals[i][0]
        cropped_voltage.append(voltage)
        cropped_current.append(current)

    a, b = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t), cropped_voltage, cropped_current, p0=(3, 1e-4))

    fitted_data = []
    for i in range(0, len(cropped_current)):
        fitted_data.append(a[0]*np.exp(a[1]*cropped_voltage[i]))

    print a[0], a[1]
    plt.plot(cropped_voltage, fitted_data, label='fit')
    plt.plot(cropped_voltage, cropped_current, label='data')
    plt.legend()
    plt.title('Original Cropped + Fitted Data')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.show()

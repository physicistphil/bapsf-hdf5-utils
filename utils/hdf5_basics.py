import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import scipy
import matplotlib.animation as animation
from scipy.signal import argrelextrema
import sys

import h5Parse as h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
BASIC DIGITAL SIGNAL PROCESSING
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
    signal = (signal - 2.0**15)*(5.0 / 2.0**16)
    # commented out to remove bias
    # signal = signal - np.mean(signal)
    return signal

def highpass_filter(signal, cutoff_freq=10000, sampling_rate=6.25e6):
    """
    PURPOSE
    -------
        This function is a high pass filter for an inputted time signal.
    
    INPUTS
    ------
        signal: 1d array
            1d time array that will be filtered.
        
        cutoff_freq: integer (optional)
            The cutoff frequency, all frequencies below this will be removed.
        
        sampling_rate: float (optional)
            The sampling rate of the inputted signal, also called
            "clockRate".
    
    OUTPUTS
    -------
        filtered_signal: 1d array
            The high pass filtered 1d array time signal.
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

def lowpass_filter(signal, cutoff_freq=10000, sampling_rate=6.25e6):
    """
    PURPOSE
    -------
        This function is a low pass filter for an inputted time signal.
    
    INPUTS
    ------
        signal: 1d array
            1d time array that will be filtered.
        
        cutoff_freq: integer (optional)
            The cutoff frequency, all frequencies above this will be removed.
        
        sampling_rate: float (optional)
            The sampling rate of the inputted signal, also called
            "clockRate".
    
    OUTPUTS
    -------
        filtered_signal: 1d array
            The low pass filtered 1d array time signal.
    """
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data)
        return y

    filtered_signal = butter_lowpass_filter(signal, cutoff_freq, sampling_rate)
    return filtered_signal

def calculate_fft(signal, sampling_rate=6.25e6, plot=False):
    """
    PURPOSE
    -------
        This function takes a signal and calculates its power
        spectrum, fft, and frequencies from the FFT.
    
    INPUTS
    ------
        signal: 1d array
            1d time array of desired data to be FFTed. 
            
        sampling_rate: float (optional)
            The sampling rate of the inputted signal, also called
            "clockRate".
        
        plot: Boolean (optional)
            Set to default as False, if True then the function plots the power spectrum
            vs. log10 of frequency. 

    OUTPUTS
    -------
        ps: 1d array
            1d array of the power spectrum of the inputted time signal in the 
            frequency domain.
        
        fft: 1d array
            The fft of the inputted signal.
            
        frequencies: 1d array
           The corresponding frequencies from the FFT.
        
        Plot of the power spectrum vs. log10 frequency of signal.
    """
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    time_step = 1.0/sampling_rate
    ps = np.abs(np.fft.fft(signal))**2
    fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(ps.size, time_step)
    
    if plot == True:
        plt.plot(np.log10(frequencies), ps)
        plt.title('Power Spectrum of Signal')
        plt.xlabel('Log10(Frequency) in Hz')
        plt.ylabel('Amplitude')
        plt.show()

    return ps, fft, frequencies
    
def correlation_function(signal1, signal2, sampling_rate=6.25e6, delta_x=2.875):
    """
    PURPOSE
    -------
        This function calculates the correlation function of the two
        inputted probe data.

    INPUTS
    ------
        signal1: 1d array
            The 1d array of first signal.
        
        signal2: 1d array
            The 1d array of second signal.
        
        sampling_rate: float (optional)
            The sampling rate of the inputted signal, also called
            "clockRate".

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
    correlation = np.correlate(signal1, signal2,'full')
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
    
def find_nearest(array, value):
    """
    PURPOSE
    -------
        This function finds the nearest element to the inputted value in 
        the array.

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
        This function determines the highest value
        local maxima of an inputted array along with 
        the index where it is located.

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
OPEN HDF5 DATA
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
            Number of shots 
        
        gain: integer (optional)
            The gain on the voltage
            
        resistance: integer (optional)
            The resistance on the current data
    
    OUTPUTS
    -------
        voltage_array: 3d array
            Voltage array in volts formatted as: (position, shot num, time signal)
        
        current_array: 3d array
            Current array in Amperes formatted as: (position, shot num, time signal)
    """
    print("CHOOSE VOLTAGE ARRAY:")
    print("......................")
    voltage = h.openHDF5_dataset(f)['data']
    print("CHOOSE CURRENT ARRAY")
    print("......................")
    current = h.openHDF5_dataset(f)['data']
    
    voltage_list = [voltage[num_shots*i:num_shots*i+num_shots] for i in range(0, 
        int(math.ceil(voltage.shape[0])/num_shots))]
    current_list = [current[num_shots*i:num_shots*i+num_shots] for i in range(0, 
        int(math.ceil(current.shape[0])/num_shots))]
    voltage_array = np.array(voltage_list)
    current_array = np.array(current_list)

    voltage_array = gain*bit_to_voltage(voltage_array)
    current_array = -1*(bit_to_voltage(current_array)/resistance)

    return  voltage_array, current_array

def get_motion_list(f):
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

    OUTPUTS
    -------
        d: dictionary
            A dictionary of the specified motion list.
    """
    dsKeys = list(f['Raw data + config']['6K Compumotor'].keys())
    dsets = f['Raw data + config']['6K Compumotor']
    for it in range(len(dsKeys)):
        print('{}  : {}'.format(it, dsKeys[it]))
    print("..........................")
    choice = str(eval(input("\n Choose a dataset: \n")))
    if choice.isdigit():
        if int(choice) in range(len(dsKeys)):
            dset = dsets[dsKeys[int(choice)]]
            if int(choice) < 3:
                d = {}
                attrs = list(dset.attrs.values())
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
            print("Dataset not found...")
            return -1
    else:
        print("Not a number.")
        return -1

def generalized_data(f, num_shots):
    """
    PURPOSE
    -------
        This function is the most general format way of pulling out
        the desired data out of the HDF5 file.
        NOTE: This does not account for the voltage amplification, need to
        multiply the signal by the gain!

    INPUTS
    ------
        f: hdf5 file
            The hdf5 file where the Langmuir data is located
            
        num_shots: integer
            Number of shots 

    OUTPUTS
    -------
       data_points: list of lists
            A list of list in the format: [[x, y, shot number, time signal], ...]
    """
    data_struct = h.openHDF5_dataset(f)
    data = data_struct['data']
    dataType = data_struct['data type']
    
    motion_list = get_motion_list(f)
    x_motion = motion_list['x']
    y_motion = motion_list['y']

    data_points = []
    shot = 0
    for i in range(0, data.shape[0]):
        signal = bit_to_voltage(data[i])
        data_points.append([x_motion[i], y_motion[i], shot, signal])
        shot += 1
        shot %= num_shots
        if i % (data.shape[0] // 1000) == 0:     # avoid excessive print statements
            print("\rLoading... {0:.2f}".format(i * 100.0 / data.shape[0]), "%", end= '', flush=True) 
    print("\rLoading... {0:.2f}".format(100.0), "%")

    return data_points

def generalized_reshaped_data(f, num_shots, Nx, Ny):
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

        Nx: float
            Number of x points

        Ny: float
            Number of y points

    OUTPUTS
    -------
        array: 4d array
            4d array of (y, x, num_shots, time signal)
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
FINDING TEMPERATURE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def temperature_curve_fit(voltage, current, vmax=20):
    """
    PURPOSE
    -------
        This function goes hand in hand with the function 
        calculate_temperature_profile. This function curve fits
        the portion of voltage and current plot to an exponential
        and returns the coeffients of the fitted curve.

    INPUTS
    ------
        voltage: 1d array
            A 1d array of 1 pulsed signal of the voltage.

        current: 1d array
            A 1d array of the same 1 pulsed signal of the current.

        vmax: integer or float (optional)
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

def calculate_temperature_profile(f, plot=True, vmax=20, cutoff_min=46339, cutoff_max=49363, num_shots=15, 
    gain=50, resistance=110):
    """
    PURPOSE
    -------
        This function averages the 15 shots for both the current and voltage
        sweep data then curve fits an exponential to the V vs. I plot to
        calculate the temperature profile of the plasma. This function utilizes cutoff, 
        this is because there are multiple pulses in the voltage/current array. The temperature
        is based off one of the pulses, but code can be changed to average the temperatures
        for all the pulse, just add another for loop.

    INPUTS
    ------
        file: HDF5 file
            The desired HDF5 file with the data.

        plot: Boolean (optional)
            If True then function plots the temperature vs. position. 

        vmax: integer or float (optional)
            A value set to be the upper cutoff for the curve.
        
        cutoff_min: integer (optional)
            The minimum index of the voltage array, set as the lower cutoff.
        
        cutoff_max: integer (optional)
            The maximum index of the voltage array, set as the upper cutoff

        num_shots: integer (optional)
            Number of shots 
        
        gain: integer (optional)
            The gain on the voltage
            
        resistance: integer (optional)
            The resistance on the current data

    OUTPUTS
    -------
        A_list: list
            List of the coeffienct A of the fitted curve. I = Aexp(B*V)
    
        temp_list: list
            List of the temperatures of the fitted curve.
    """
    voltage, current = data_parser_langmuir(f, num_shots=15, gain=50, resistance=110)
    voltages = np.mean(voltage, axis=1)
    currents = np.mean(current, axis=1)

    A_list = []
    temp_list = []
    for i in range(0, voltage.shape[0]):
        voltage, current = voltages[i][cutoff_min:cutoff_max], currents[i][cutoff_min:cutoff_max]
        a, b = temperature_curve_fit(voltage, current, vmax=vmax)
        A_list.append(a)
        temp_list.append(1/b)

    if plot == True:
        plt.plot(temp_list)
        plt.title('Temperature as a function of position')
        plt.xlabel('Position (index)')
        plt.ylabel('Temperature (eV)')
        plt.show()

    return temp_list

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PLOTTING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def basic_2d_plot(data, time_idx, xmin, xmax, ymin, ymax, delta_x, delta_y, individual_shot=False, 
    shot_num = 0, contour=True):
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
    
        xmin: float
            The minimum position in cm of the x axis.

        xmax: float
            The maximum position in cm of the x axis.

        ymin: float
            The minimum position in cm of the y axis.

        ymax: float
            The maximum position in cm of the y axis.

        delta_x: float
            The change in the x positions in cm.

        delta_y: float
            The change in the y positions in cm.

        individual_shot: Boolean (optional)
            False = data has been averaged over all shots.
            True = pick a shot number to animate that data.
    
        shot_num: integer (optional)
            If individual_shot = True, then specify which shot to animate. Default is 0.

        contour: Boolean (optional)
            If True, then function overlays the contour plot.

    OUTPUTS
    -------
        2d plot of the data at one instance in time
    """
    if individual_shot == False:
        plt.imshow(data[:, :, time_idx], extent=[xmin, xmax, ymin, ymax], interpolation='catrom')
        plt.colorbar()
        if contour == True:
            x = np.arange(xmin, xmax, delta_x)
            y = np.arange(xmin, xmas, delta_y)
            plt.contour(x, y, data[:, :, time_idx])
    else:
        plt.imshow(data[:, :, shot_num, time_idx], extent=[xmin, xmax, ymin, ymax], interpolation='catrom')
        plt.colorbar()
        if contour == True:
            x = np.arange(xmin, xmax, delta_x)
            y = np.arange(xmin, xmax, delta_y)
            plt.contour(x, y, data[:, :, shot_num, time_idx])
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    #plt.title('ADD YOUR TITLE HERE')
    plt.show()
    
def animate_time_signal(data, interval=30, individual_shot=False, shot_num=0):
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
        im = plt.imshow(data[:, :, 0], interpolation='catrom')
    else:
        im = plt.imshow(data[:, :, shot_num, 0], interpolation='catrom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()

    def updatefig(j):
        if individual_shot == False:
            im.set_array(data[:, :, j])
        else:
            im.set_array(data[:, :, shot_num, j])
        return [im]

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
        data: 3d or 4d array
            A 3d or 4d array meant to be formatted as: (x, y, time) or (x, y, shot_num, time)

        u: 3d array or 4d array
            The x component of the magnetic field (vector field). Should be in the shape 
            of (x, y, time) or (x, y, shot_num, time).

        v: 3d array or 4d array
            The y component of the magnetic field (vector field). Should be in the shape 
            of (x, y, time) or (x, y, shot_num, time).
            
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
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
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

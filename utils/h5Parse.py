import h5py
import numpy as np
#import CH_analysis as CH
import matplotlib.pyplot as plt

# written for python 2.7.5
# CODE IS VERY MUCH IN PROGRESS...JUDGE ME NOT - Jeff

# watch out for 'CHECK THIS' inserted into comments...code may not be correct

# when reshaping numpy arrays, use the Fortran index ordering 'F'. This seems to be the correct ordering (as opposed to C)
# despite the inversion of the indices in the datasets.

# -----------------------------------------------------------------------------

def DAQconvert(data,digi_type,offset=0.0) :
  # digi_type identifies the type of digitizer used. Currently only supports the 100 MHz and 1.25 GHz digitizers
  # offset is in volts
  digitizerStats = {'3302':[16,5.062],'3305':[12,2.0]} # First item of each key is the bit number of the digitizer, second is the voltage range

  if digi_type in digitizerStats :
    output = (data/2.0**(digitizerStats[digi_type][0])-.5)*digitizerStats[digi_type][1] + offset
  else :
    print("Digitizer type specified in 'DAQconvert' does not correspond to an available digitizers")
    print("please select from the following digitizers: \n",list(digitizerStats.keys()))
    output = data.copy()
  return output

def expandDims(dataset_shape,motionList) : # - written 8/10/2016
  """
    Takes a dataset_shape and a motion list (an item in the list of the output of get_MotionLists and outputs a shape representative of the probe motion

    ------------------------
    inputs:
    ------------------------ 
    dataset_shape : tuple      : The size of the dataset that will eventually be reshaped to match the motion list
    motionList    : dictionary : The output to get_MotionLists that corresponds to the probe motion list
    
    ------------------------    
    output:
    ------------------------
    returns tuple of the shape of a dataset that was taken with the input motion list

    ------------------------
    use :
    ------------------------
    If 'testFile.hdf5' contains a dataset with id 'd_id' and 1 motion list, the easiest call is

      result = expandDims(d_id.shape,get_MotionLists('testFile.hdf5')[0])

    If get_MotionLists('testFile.hdf5')[0] is an xy plane (nx=11,ny=12) where each position has 10 shots and 1024 times and the dataset_shape is given as (1320,1024)
    the output is a tuple, (12,11,10,1024)

    If multiple motion lists are in the HDF5 file, one has to know the proper index of the output to get_MotionLists.

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 8/10/2016 - Jeffrey Bonde

    ------------------------
    to-do list :
    ------------------------ 
    implement a way to request a motion list known to be in the corresponding file
  """

  shots = dataset_shape[0]
  times = dataset_shape[1]
  compatible = False
  type = motionList['type']
  
  output_shape = ()
  size = 1
  # Zs and Ys move last. CHECK THIS: if ever there is a 3D probe drive or a 'ZY' type of plane, the ordering here must be checked to ensure last moving dimension is assigned first to the output_shape
  # If anything changes slower than the Z or Y motion, a custom size must be put in and you can't use this method to get the correct reshaping of the dataset
  if 'Z' in type :
    nz = motionList['nz']
    output_shape = output_shape + (nz,)
    size *= nz
  if 'Y' in type :
    ny = motionList['ny']
    output_shape = output_shape + (ny,)
    size *= ny
  if 'X' in type :
    nx = motionList['nx']
    output_shape = output_shape + (nx,)
    size *= nx
  nshots = motionList['nshots']
  output_shape = output_shape + (nshots,)
  size *= nshots

  output_shape = output_shape + (times,)

  if size != shots :
    print("Could not expand dimensions to match number of elements in dataset_shape using the given motionList. Returning dataset_shape")
    return dataset_shape
  else :
    return output_shape

def extractData(dataset_id, data_shape=None, hyperslab='') : # - written 8/10/2016
  """
    extracts data from the given dataset and reshapes it to 'shape' with optional hyperslab extraction

    ------------------------
    inputs:
    ------------------------ 
    dataset_id : HDF ID tag : The id corresponding to the HDF5 dataset that is to be extracted
    data_shape : tuple      : The desired or intended output shape of the dataset. If 'None' no reshaping on the data is done.
    hyperslab  : str        : a string representation of bracket selection of a subset of data. See 'use' below
    
    ------------------------    
    output:
    ------------------------
    returns numpy.array of the data stored in the dataset

    ------------------------
    use :
    ------------------------
    If 'testFile.hdf5' contains a dataset with id 'd_id' and shape (20,1000) that is to be reshaped into a (2,10,1000) numpy.array, the proper call is

      result = extractData(d_id,(2,10,1000))

    If one only wants the values corresponding to indices [2,3,4,5] in the 2nd dimension of the output array, using the hyperslab parameter,

      result = extractData(d_id,(2,10,1000),'[:,2:6,:]')

    This call has an identical output to
    
      result = extractData(d_id,(2,10,1000))[:,2:6,:]

    but uses less memory and is faster for large datasets.

    The data_shape parameter is meant to be associated with the motion list corresponding to the dataset.
    An example of the full call if the motion list 'ML' is the one corresponding to the dataset is

      result = extractData(d_id,expandDims(d_id.shape,ML))

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 8/10/2016 - Jeffrey Bonde
      NOTES: Eh, it seems to work.

    ------------------------
    to-do list :
    ------------------------ 
    implement a way to request a motion list known to be in the corresponding file
  """
  if data_shape == None :
    data_shape = dataset_id.shape

  if hyperslab=='' : # hyperslab not given, extract everything
    output = np.reshape(dataset_id[:],data_shape)
  else :
    sliceStrs = HyperslabToSliceStrings(hyperslab)
    if len(sliceStrs) != len(data_shape) :
      print("Hyperslab dimension does not correspond to data_shae, returning dataset_id")
      output = dataset_id
    else :
      # I would like to be able to handle the time and shot index separately...but apparently that's not supported yet by h5py. So I'm stuck with creating a hyperslab array over all dimensions
      # last string should be for time...want to handle that carefully
      if len(sliceStrs) > 1 :
        shotSliceStrs = sliceStrs[0:-1]
        timeSliceStr = sliceStrs[-1]

        shotTuple = SliceStringsToSliceTuple(shotSliceStrs,data_shape[0:-1])
        shotIndices = NDSliceToLinear(shotTuple,data_shape[0:-1])
        extractionShape = ()
        for iter in shotTuple :
          dim = len(iter)
          if dim != 1 :
            extractionShape = extractionShape + (len(iter),)
        if timeSliceStr == ':' :
          output = np.reshape(dataset_id[shotIndices,:],extractionShape+(data_shape[-1],))
        else : 
          timeTuple = SliceStringsToSliceTuple([timeSliceStr],(data_shape[-1],))
          timeIndices = timeTuple[0]
          # CHECK THIS: h5py does not yet have support for multiple vector selection so I have to hyperslab one dimension and select the other dimension later. For speed I have to choose the order that loads in the least data first.
          if len(timeIndices)*dataset_id.shape[0] < dataset_id.shape[1]*len(shotIndices) :
            temp = dataset_id[:,timeIndices]
            temp = temp[shotIndices]
          else :
            temp = dataset_id[shotIndices,:]
            temp = temp[:,timeIndices]
          if len(timeIndices) != 1 :
           extractionShape = extractionShape + (len(timeIndices),)
          output = np.reshape(temp,extractionShape)


      else :
        timeSliceStr = sliceStrs
        if timeSliceStr == ':' :
          output = dataset_id[:] # no reshaping needed
        else : 
          timeTuple = SliceStringsToSliceTuple([timeSliceStr],data_shape[-1])
          timeIndices = timeTuple[0]
          output = dataset_id[timeIndices]
  return output

# convert a hyperslab to slice Strings
def HyperslabToSliceStrings(hyperslab) : # - written 8/10/2016
  return hyperslab.strip('[]').split(',')

def SliceStringsToSliceTuple(sliceStrings,forShape=None) : # - written 8/10/2016
  """
  User inputs sliceString which is a string representation of the desired selection from some array.
  This converts the string to a tuple of lists of indices with the same shape

  For example:
  if sliceStrings = ['4:6','3','2:4']
   SliceStringsToSliceTuple(sliceString) outputs
   ([4,5],[3],[2,3])
  
   if sliceStrings contains a terminal ':' in a dimension, forShape must be specified
   e.g. SliceStringsToSliceTuple(['4:6',':','0'],(6,3,2)) outputs
   ([4,5],[0,1,2],[0])
  """
  output = () # empty tuple
  for it in range(len(sliceStrings)) :
    s_it = sliceStrings[it]
    if ':' in s_it :
      ncol = s_it.count(':')
      if ncol == 1 : # unstepped slice
        step = 1
        loc1 = s_it.find(':')
        if loc1 == len(s_it)-1 :
          stop = forShape[it]
        else :
          stop = int(s_it[loc1+1:])
        
        if loc1 == 0 :
          start = 0
        else :
          start = int(s_it[0:loc1])
        output = output +(list(range(start,stop,step)),)
      elif ncol == 2: # stepped slice
        loc1 = s_it.find(':')
        loc2 = s_it.rfind(':')
        if loc2 == len(s_it)-1 :
          step = 1
        else :
          step = int(s_it[loc2+1:])
        stop = int(s_it[loc1+1:loc2])
        if loc1 == 0 :
          start = 0
        else :
          start = int(s_it[0:loc1])
        output = output +(list(range(start,stop,step)),)
      else : 
        print("improper slicing request") # should probably output an error rather than output a tuple with a valid index...
        return ([-1],)
    else :
      output += ([int(s_it)],)
  return output

# each input is a tuple
def NDSliceToLinear(slice_tuple,forShape) : # - written 8/10/2016
  """
  Takes an input slice_tuple, which is a tuple of lists of indices, and converts it to a single list of linear indices

  For example. If working with an object with shape (4,3,2) and I want the indices corresponding to the 4 points ([1,2],[0,1],[1]),
  the input is ([1,2],[0,1],1]) and the output is
  [7, 9, 13, 15] for an object with shape (24,)

  Since the HDF5 stores an array of size (Nx*Nz*Nshots,Nt) where Nx are the number x positions, Nz - z positions, Nshots - shots at each position, Nt - number of times, which corresponds to a real array of shape (Nz, Nx, Nshots, Nt),
  this is the conversion if I want to grab the set of z positions, x positions, and shots
  """
  maxPrevSize = 1
  nd_slice = len(slice_tuple)
  temp = slice_tuple[nd_slice-1]
  
  for it in range(nd_slice-2,-1,-1) :
    prev = temp[:]
    temp = []
    maxPrevSize *= forShape[it+1]
    for jt in range(len(slice_tuple[it])) :
      temp += [x+slice_tuple[it][jt]*maxPrevSize for x in prev]
  return temp

def get_MotionLists(file) : # grab motion list data - Written 8/9/2016
  """
    extracts motion lists from the HDF5 file

    ------------------------
    inputs:
    ------------------------ 
    file : str : the filename of the HDF5 file from which to extract the dataset
    
    ------------------------    
    output:
    ------------------------
    returns a list of dictionaries containing the key data from each of the motion lists

    current keys in each dictionary are:
      'Nz'      : integer : integer number of z-positions in the motion list. Most programs below will ignore this if Nz=1.
      'Ny'      : integer : integer number of y-positions in the motion list. Most programs below will ignore this if Ny=1.
      'Nx'      : integer : integer number of x-positions in the motion list. Most programs below will ignore this if Nx=1.
      'zmesh'   : [Nz,Ny,Nx] numpy.array : z component of the meshgrid of the motion list. Dimensions of size 1 are ignored.
      'ymesh'   : [Nz,Ny,Nx] numpy.array : y component of the meshgrid of the motion list. Dimensions of size 1 are ignored.
      'xmesh'   : [Nz,Ny,Nx] numpy.array : x component of the meshgrid of the motion list. Dimensions of size 1 are ignored.
      'nshots'  : integer : number of shots taken at each position. Motion lists are written with the total number of shots taken in the corresponding dataset, Nshots_total. This value is calculated as Nshots_total/(Nx*Ny*Nz).
      'type'    : str : The type of geometry that the motion list corresponds to. For example 'XY' is an xy-plane. 'X' is an x-line. '' is no motion (Nz=Ny=Nx=1).
      'portNum' : integer : the port number location where the motion took place. This is incorrect if multiple drives use the same motion list.
      'name'    : str : The name of the motionlist

    ------------------------
    use :
    ------------------------
    If 'testFile.hdf5' has two motion lists where 'ml1' and 'ml2' are the dictionaries each 
    containing the above keywords for their associated motion lists, then

    result = get_MotionLists('testFile.hdf5')

    contains

    [ml1, ml2]

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 8/9/2016 - Jeffrey Bonde
      NOTES: Eh, it seems to work.

    ------------------------
    to-do list :
    ------------------------ 
    implement a way to request a motion list known to be in the corresponding file
  """
  dict_supportedDrives = {'6K Compumotor':get_6KMotionList, 'Velmex_XZ':get_VelmexMotionList, 'NI_XZ':get_NI_XZMotionList}
  # motion list is going to be in the 'Raw data + config' directory in the HDF5 file
  motionLists = []
  if 'Raw data + config' in list(file.keys()) :
    rdc = file['Raw data + config']
    for iter in rdc : 
      if iter in dict_supportedDrives :
        motionLists += dict_supportedDrives[iter](rdc[iter])
  else : 
    print("Normal motion list location not found, returning empty Motion List")
    
  return motionLists

def get_NI_XZMotionList(dir_drive) : #  - Written 8/10/2016
  # dir_drive is the (opened) NI_XZ drive directory which contains the run time list and configuration folder
  # open up the run time list to get the proper motion list directory
  runTime = dir_drive['Run time list'] # open the drive run time list
  configs = runTime['Configuration name'] # retrieve all motion list configuration names for Velmex drive
  Nconfigs = configs.size
  if (configs == configs[0]).all() : # if all configuration names are the same
    config = configs[0]
    if config in dir_drive :# Found configuration name in the drive directory
      print("opened configuration")
      nx = dir_drive[config].attrs['Nx']
      nz = dir_drive[config].attrs['Nz']
      dx = dir_drive[config].attrs['dx']
      dz = dir_drive[config].attrs['dz']
      x0 = dir_drive[config].attrs['x0']
      z0 = dir_drive[config].attrs['z0']

      type = ''
      if nx > 1 :
        type +='X'
      if nz > 1 :
        type +='Z'

      xs = x0 + np.arange(nx)*dx
      zs = z0 + np.arange(nz)*dz

      xmesh, zmesh = np.meshgrid(xs, zs)
      ny = 1
      ymesh = np.zeros_like(xmesh,dtype=float)

      nshots = Nconfigs/xmesh.size

      portNum = dir_drive[config].attrs['z_port']

      motionList = [{'xmesh':xmesh,'ymesh':ymesh,'zmesh':zmesh,'nx':nx,'ny':ny,'nz':nz,'nshots':nshots,'type':type,'portNum':portNum,'name':config}]
    else : # did not find configuration name in the drive directory
      print("configuration in the run time list does not match any available configuration files in drive folder")
      motionList = [{'empty':-1}]
  else : # CHECK THIS. I do not yet know how to handle when the Velmex sees more than one motion list in the run time list
    print("incomplete part of code reached. I cannot handle multiple motion lists in a Velmex run time list")
    motionList = [{'empty':-1}]
  return motionList

def get_6KMotionList(dir_drive) : #  - Written 8/9/2016
  # dir_drive is the (opened) 6K Compumotor directory which contains the run time lists and configuration folders
  # open up the run time list to get the proper motion list directory
  # the 6K Compumotor can handle multiple motion lists.  Have to parse them
  motionLists = []
  for iter in dir_drive :
    if isinstance(dir_drive[iter],h5py.Dataset) :
      temp = format6KmotionList(dir_drive,iter)
      # CHECK THIS...I also want to not add duplicate motion lists to the output but for that I would have to compare two outputs of format6KmotionList. That is, I'm going to have to make a motion list object
      if 'empty' not in temp :
        motionLists += [temp]
  return motionLists

def format6KmotionList(dir_drive, mL_dset) : #  - Written 8/9/2016
  configs = dir_drive[mL_dset]['Motion list'] # retrieve all motion list configuration names for 6K Compumotor drive
  probeConfig = 'Probe: '+str(mL_dset) 
  if probeConfig in dir_drive :
    portNum = dir_drive[probeConfig].attrs['Port']
  else :
    print("Couldn't find port number in configuration file for probe corresponding to motion list: "+str(ml_dset))
    portNum = -1
  if (configs == configs[0]).all() : # if all configuration names are the same
    mlName = configs[0]
    config = 'Motion list: '+configs[0] # don't know why Velmex doesn't have to do this
    if config in dir_drive : # configuration file exists
      
      nx = dir_drive[config].attrs['Nx']
      ny = dir_drive[config].attrs['Ny']
      nz = 1

      dx = dir_drive[config].attrs['Delta x']
      dy = dir_drive[config].attrs['Delta y']
      
      xc = dir_drive[config].attrs['Grid center x']
      yc = dir_drive[config].attrs['Grid center y']

      x0 = xc - (nx-1)/2.0*dx
      y0 = yc - (ny-1)/2.0*dy

      xs = np.arange(nx)*dx+x0
      ys = np.arange(ny)*dy+y0

      type = ''
      if nx > 1 :
        type +='X'
      if ny > 1 :
        type +='Y'

      x_list = dir_drive[mL_dset]['x']

      Nmotions = nx*ny

      nshots = x_list.size/Nmotions

      xmesh, ymesh = np.meshgrid(xs, ys)
      zmesh = np.zeros_like(xmesh,dtype=float)
      portNum = -1 # CHECK THIS...have to fix. The actual port number is listed in the probe config file
      
      motionList = {'xmesh':xmesh,'ymesh':ymesh,'zmesh':zmesh,'nx':nx,'ny':ny,'nz':nz,'nshots':nshots,'type':type,'portNum':portNum,'name':mlName}
    else :
      print("configuration file--" + config + "--for motion list--"+dir_drive[mL_dset].name+"--not found")
      motionList = {'empty':-1}
  else : # CHECK THIS. I do not yet know how to handle when the 6K Compumotor sees more than one motion list in the run time list for a given probe
    print("incomplete part of code reached. I cannot handle multiple motion lists for one probe run time list")
    motionList = {'empty':-1}
  return motionList

def get_VelmexMotionList(dir_drive) : #  - Written 8/9/2016
  # dir_drive is the (opened) Velmex Drive directory which contains the run time list and configuration folder
  # open up the run time list to get the proper motion list directory
  runTime = dir_drive['Run time list'] # open the drive run time list
  configs = runTime['Configuration name'] # retrieve all motion list configuration names for Velmex drive
  Nconfigs = configs.size
  if (configs == configs[0]).all() : # if all configuration names are the same
    config = configs[0]
    if config in dir_drive :# Found configuration name in the drive directory
      x_list = dir_drive[config].attrs['Velmex_XZ_x_list'][:] # why the there is an underscore in one and not the other I will never know
      z_list = dir_drive[config].attrs['Velmex_XZ_z list'][:]
      nx = dir_drive[config].attrs['nx']
      nz = dir_drive[config].attrs['nz']

      type = ''
      if nx > 1 :
        type +='X'
      if nz > 1 :
        type +='Z'

      xmesh = np.reshape(x_list,(nx,nz),'F').T
      zmesh = np.reshape(z_list,(nx,nz),'F').T
      ymesh = np.zeros((nx,nz),dtype=float)

      nshots = Nconfigs/xmesh.size

      portNum = dir_drive[config].attrs['LAPD Port number']
      ny = 1
      motionList = [{'xmesh':xmesh,'ymesh':ymesh,'zmesh':zmesh,'nx':nx,'ny':ny,'nz':nz,'nshots':nshots,'type':type,'portNum':portNum,'name':config}]
    else : # did not find configuration name in the drive directory
      print("configuration in the run time list does not match any available configuration files in drive folder")
      motionList = [{'empty':-1}]
  else : # CHECK THIS. I do not yet know how to handle when the Velmex sees more than one motion list in the run time list
    print("incomplete part of code reached. I cannot handle multiple motion lists in a Velmex run time list")
    motionList = [{'empty':-1}]

  return motionList

 #  - Written 8/9/2016
def get_MSI(file,choice='') : # list available MSI options or allow user to programmatically ask for a known choice of MSI data
  # A dictionary of the functions that are currently written to extract MSI data
  dict_supportedMSIdata = {'Discharge':get_MSI_discharge, 'Gas pressure':get_MSI_gas_pressure, 'Interferometer array':get_MSI_interferometers, 'Magnetic field':get_field_profile}

  # case of no choice given
  if len(choice) == 0 : 
    print("\nList of supported MSI dataset:")
    for option in list(dict_supportedMSIdata.keys()) :
      print(option)
    selection = eval(input("\nInput an option:\n"))

    # check if user input is a string (or unicode)
    if isinstance(selection,str) : 
      return get_MSI(file,selection)
    else :
      return get_MSI(file,'-1') # will return -1. Has unnecessary steps but already outputs the proper error.

  # case of choice given
  else :
    # choice is invalid. Print warning and return -1
    if choice not in dict_supportedMSIdata :
      print("user input in call 'get_MSI( HDF5_file_name , MSI choice )' is not a supported MSI choice. Returning -1.")
      return -1

    #
    else :                                    # case of valid choice
      return dict_supportedMSIdata[choice](file)

def get_MSI_discharge(file) : # grab machine state information #  - Written 8/8/2016
  if 'MSI' in file : 
    dirMSI = file['/MSI']
    if 'Discharge' in dirMSI : 
      discharge_ds = dirMSI['Discharge']
      t0 = discharge_ds.attrs['Start time']
      dt = discharge_ds.attrs['Timestep']
      I = discharge_ds['Discharge current'][:] # already in units of Amperes
      V = discharge_ds['Cathode-anode voltage'][:] # already in units of Volts

      if len(I.shape) != 1 :
        Nt = I.shape[1]
      else : 
        Nt = I.shape[0]

      ts = t0 + dt*np.arange(Nt*1.0)
    else :
      print('Discharge data missing from the MSI directory of the HDF5 file...returning -1s')
      ts = -1
      I = -1
      V = -1
  else : 
    print('MSI missing from HDF5 file...returning -1s')
    ts = -1
    I = -1
    V = -1
  return {'ts':ts, 'Current':I, 'Voltage':V}

def get_MSI_gas_pressure(file) : # grab machine state information #  - Written 8/8/2016
  if 'MSI' in file : 
    dirMSI = file['/MSI']
    if 'Gas pressure' in dirMSI : 
      gasPressure_ds = dirMSI['Gas pressure']
      masses = gasPressure_ds.attrs['RGA AMUs'] # masses of detected ions in AMU
      pressures = gasPressure_ds['RGA partial pressures'][:] # partial pressures in Torr
    else :
      print('Gas pressure data missing from the MSI directory of the HDF5 file...returning -1s')
      masses = -1
      pressures = -1
  else : 
    print('MSI missing from HDF5 file...returning -1s')
    masses = -1
    pressures = -1
  return {'masses':masses, 'pressures':pressures}

def get_MSI_interferometers(file) : # grab machine state information #  - Written 8/8/2016
  if 'MSI' in file : 
    dirMSI = file['/MSI']
    if 'Interferometer array' in dirMSI : 
      dir_Interfs = dirMSI['Interferometer array']

      numInterfs = dir_Interfs.attrs['Interferometer count']
      it = 0
      interf_iter = get_interferometer(dir_Interfs['Interferometer ['+str(it)+']'])
      # get number of time samples and shots so that the array size can be allocated
      Nt = interf_iter['ts'].size
      Nshots = interf_iter['nshots']

      # allocate
      n_bar_L = np.empty(numInterfs,dtype=float)
      zs = np.empty(numInterfs,dtype=float)
      ts = np.empty((numInterfs, Nt),dtype=float)
      interfs = np.empty((numInterfs,Nshots,Nt),dtype=float)

      # assign first set of values
      n_bar_L[it] = interf_iter['nbarl']
      zs[it] = interf_iter['z'] # z-locations in 'cm'
      ts[it,:] = interf_iter['ts']
      interfs[it,...] = interf_iter['interf']
      
      
      for it in range(1,numInterfs) : 
        # open an individual interferometer
        interf_iter = get_interferometer(dir_Interfs['Interferometer ['+str(it)+']'])
        n_bar_L[it] = interf_iter['nbarl']
        zs[it] = interf_iter['z'] # z-locations in 'cm'
        ts[it,:] = interf_iter['ts']
        interfs[it,...] = interf_iter['interf']

    else :
      print('Interferometer array data missing from the MSI directory of the HDF5 file...returning -1s')
      n_bar_L = -1
      zs = -1
      ts = -1
      interfs = -1
  else : 
    print('MSI missing from HDF5 file...returning -1s')
    n_bar_L = -1
    zs = -1
    ts = -1
    interfs = -1
  return {'nbarl':n_bar_L, 'zs':zs, 'ts':ts, 'interfs':interfs}

def get_interferometer(interf_name) : # retrieves an individual record of an interferometer from the MSI directory 'Interferometer array' #  - Written 8/8/2016
  n_bar_L = interf_name.attrs['n_bar_L']
  z_loc = interf_name.attrs['z location'] # z-locations in 'cm'
  t0 = interf_name.attrs['Start time']
  dt = interf_name.attrs['Timestep']

  interf_trace = interf_name['Interferometer trace'][:]

  if len(interf_trace.shape) != 1 :
    Nt = interf_trace.shape[1]
    Nshots = interf_trace.shape[0]
  else : 
    Nt = interf_trace.shape[0]
    Nshots = 1

  ts = ts = t0 + dt*np.arange(Nt*1.0)

  return {'nbarl':n_bar_L, 'z':z_loc, 'ts':ts, 'interf':interf_trace, 'nshots':Nshots}

def get_field_profile(file) : #  - Written 8/9/2016
  if 'MSI' in file : 
    dirMSI = file['MSI']
    if 'Magnetic field' in dirMSI : 
      field_ds = dirMSI['Magnetic field']
      zs = field_ds.attrs['Profile z locations'] # z-locations in 'cm'
      Bs = field_ds['Magnetic field profile'][:] # magnetic field profile in Gauss
      supCur = field_ds['Magnet power supply currents'][:] # current running through the magnets. This is returned as they can be used as inputs to construct a more detailed picture of the magnetic field within the LaPD.
    else :
      print('Magnetic field data missing from the MSI directory of the HDF5 file...returning -1s')
      zs = -1
      Bs = -1
      supCur = -1
  else : 
    print('MSI missing from HDF5 file...returning -1s')
    zs = -1
    Bs = -1
    supCur = -1
  return {'zs':zs, 'Bs':Bs,'Supply Currents':supCur}

def get_ListOfDatasets(file) : # grab metadata of all datasets in 'SIS crate' group
  """
    compiles a list (NTO) of datasets in the given HDF5 file that are located in standard groups for BAPSF HDF5 files

    ------------------------
    inputs:
    ------------------------    
    file : str : the filename of the HDF5 file that will be parsed

    ------------------------    
    output:
    ------------------------
    returns a dictionary of dictionaries containing information regarding each dataset

    The 1st level of dictionary has the names of the datasets found as the keys.
    The values of each key of the 1st level dictionary are dictionaries (2nd level) containing the properties (extracted from attributes) of the datasets.

    The keys of the second level dictionary are currently:
      'channel'    : str     : the channel number for the given digitizer board
      'clock rate' : float64 : the effective digitization rate of the dataset
      'data type'  : str     : the data type as supplied by the user when the data was taken
      'data'       : uint    : the values of the dataset where the bit depth is dependent on the digitizer used.
                               Note that I will not be implementing the code here to convert to appropriate floating point
      'digitizer'  : str     : an identifier for the particular digitizer used, i.e. 3305 for the FPGA or 3302 for the standard digitizer.
                               This is left in to ensure an ID gets carried with the dataset to indicate any appropriate conversions
      'dir'        : str     : a string containing the group directory of the dataset within the HDF5 file
      'size'       : tuple   : a tuple containing the shape of the dataset

    ------------------------
    use :
    ------------------------    
    Say I want to look at the datasets in 'testFile.hdf5'. This file has two datasets, 'ds1' and 'ds2'
    with shapes (3,6) and (2,5), respectively. Then I can print the names of the datasets in the file
    with the following short code:

    result = get_ListOfDatasets('testFile.hdf5')
    print result.keys()

    which prints:

    ['ds1','ds2']

    and if I want to check the size of 'ds1':

    print result['ds1']['size']

    which prints:

    (3,6)

    REMEMBER TO CLOSE THE FILE WHEN YOU ARE DONE!

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 7/21/2016 - Jeffrey Bonde
      NOTES: There's quite a bit that is hard coded that I don't like but it's more a problem with the 
      current HDF5 saving paradigm. Specifically, the groups in the hierarchy are not titled, attributed
      to or associated with what hardware. This means, directories in which to look for motion lists and
      datasets must be hard coded based on available hardware. This is a very crappy 1st draft and I 
      have not spent much time making this robust. It does seem to work.

    update: 8/4/2016 - Jeffrey Bonde    
      fixed a pretty obvious bug in how the digitizer configuration files are linked with the datasets.

    ------------------------
    to-do list :
    ------------------------ 
    make 'meta' dictionary an object that can be instantiated as null or with a dataset.attrs dictionary
      Right now, every time a property is added, all instances should (don't necessarily have to)
      be changed.
  """

  dsets = {} # output dictionary of datasets

  # Check for 'Raw Data + config' directory
  if 'Raw data + config' in list(file.keys()) :
    rdc = file['Raw data + config']
    data_run_description = rdc.attrs['Description'] # description of the data run written by user in LabView file

    # check for 'SIS crate' directory
    if 'SIS crate' in list(rdc.keys()) :
      sis_crate = rdc['SIS crate'] # open the 'SIS crate' group where datasets are normally found

      # lists to keep track of. Not all are really necessary.
      datasetNames = []  # list of keys that are datasets
      datasetDirs = []   # list of directories corresponding to each dataset
      digitizerType = [] # list of digitizer numbers corresponding to each dataset
      channel = []       # channels of the datasets
      boardName = []     # names of the board of the digitizer corresponding to the datasets
      clockRate = []     # clock rates for the datasets
      dataType = []      # data type of the datasets
      datasetShapes = [] # shapes of the datasets
      datasetIDs = []    # IDs of dataset

      for iter in sis_crate:                            # iterate over the keys
        dir_id = sis_crate[iter]
        if isinstance(dir_id,h5py.Dataset) :   # does the key correspond to a dataset?
          # check the dataset name to make sure I'm not counting header datasets
          if str(iter).find('header') == -1 :
            dsName = str(iter)                          # each dataset is named '{configGroup Name} [Slot {Slot Num}: SIS {digitizer Num} {optional digitizer id} ch {channel Num}]'
            datasetNames.append(dsName)                 # add to list of datasets in 'SIS crate'
            datasetDirs.append(str(sis_crate.name)+'/') # directory of the dataset within the HDF5
            cfGroup = dsName.partition('[')[0].rstrip() # the main digitizer configuration group corresponding to the dataset
            datasetShapes.append(dir_id.shape)
            datasetIDs.append(dir_id)

            # develop slot to config indices map
            slotNums = sis_crate[cfGroup].attrs['SIS crate slot numbers']
            configInds = sis_crate[cfGroup].attrs['SIS crate config indices']

            # retrieve channel number from dataset name
            tempChan = dsName.split('ch ')[1].split(']')[0].strip() 
            # ensure that I can convert to an integer later
            if tempChan.isdigit() : 
              channel.append(tempChan)
            else : 
              print("Non-standard channel ID found in dataset name", dsName,". Setting channel number to -1")
              channel.append('-1')

            # get slot number
            slotNum = dsName.split('Slot ')[1].split(':')[0]

            # get digitizer type. Currently SIS 3305 (FPGA) or SIS 3302
            dgType = dsName.split('SIS ')[1][0:4]       # digitizer type is given by the 4 characters after 'SIS ' in the dataset name)
            # For some reason, there is an additional board name appended to the strings identifying the FPGA channels that is not there for the 3302
            bName = dsName.split(dgType)[1].split('ch')[0].strip() # name of the board. Is empty string for 3302...should be fine
            boardName.append(bName)

            # start building configuration group directory based on digitizer type and cfGroup
            # CHECK THIS: At some point, I shouldn't hardcode the base Clock Rate. This is mostly for the FPGAs.
            if dgType == '3305' :
              configSubDir_base = cfGroup + '/SIS crate 3305 configurations'

              # grab the first index where the list of slotNums is equal to the slotNum for the dataset. Pass to configInds to get the proper configuration group number
              configSubDir = configSubDir_base + '[' + str(configInds[np.where(slotNums == int(slotNum))[0][0]]) + ']'
              # Get the datatype as indicated by user when datarun is set up
              dataType.append(sis_crate[configSubDir].attrs[(bName + ' Data type ' + tempChan).strip()])

              # handle digitization rate
              baseClockRate = 1.25*10.0**9
              # CHECK THIS
              # SIS 3305 does not have a hardware average but it can run higher sampling rate...not sure how to handle this
              clockRate.append(baseClockRate)
            elif dgType == '3302' :
              configSubDir_base = cfGroup + '/SIS crate 3302 configurations'

              # grab the first index where the list of slotNums is equal to the slotNum for the dataset. Pass to configInds to get the proper configuration group number
              configSubDir = configSubDir_base + '[' + str(configInds[np.where(slotNums == int(slotNum))[0][0]]) + ']'
              # Get the datatype as indicated by user when datarun is set up
              dataType.append(sis_crate[configSubDir].attrs[(bName + ' Data type ' + tempChan).strip()])

              # handle digitization rate
              baseClockRate = 10.0**8
              hardwareAvg = sis_crate[configSubDir].attrs['Sample averaging (hardware)']
              clockRate.append(baseClockRate/2.0**(hardwareAvg))
            else : 
              print("Couldn't identify digitizer type from dataset name: ", dsName,". Inserting -1")
              dgType = "-1"
              dataType.append('unknown')
              clockRate.append(-1)

            digitizerType.append(dgType)               
              
      if len(datasetNames) == 0 :
        print("No datasets found, returning empty list")
      for it in range(len(datasetNames)) :
        meta = {'data type':dataType[it],'dir':datasetDirs[it],'dataset_id':datasetIDs[it],'channel':channel[it],'clock rate':clockRate[it],'digitizer':digitizerType[it],'size':datasetShapes[it]}
        dsets[datasetNames[it]] = meta
    else :
      print("SIS create group not found, returning empty dataset")
      meta = {'data type':'none','dir':'none','dataset_id':None,'channel':'none','clock rate':0,'digitizer':'none','size':()}
      dsets['empty'] = meta
      # return an empty dataset
  else :
    print("Normal dataset location not found, returning empty dataset")
    meta = {'data type':'none','dir':'none','dataset_id':None,'channel':'none','clock rate':0,'digitizer':'none','size':()}
    dsets['empty'] = meta
    # return an empty dataset

  return dsets

def openHDF5_dataset(file,dataset_name='') :
  """
    extracts a dataset from an HDF5 file

    ------------------------
    inputs:
    ------------------------ 
    file : str : the filename of the HDF5 file from which to extract the dataset
    dataset_name : str : (optional) if given, searches for the named dataset and extracts it without user input
                         if not given, searches for all datasets in the digitizer group and gives a list
                         from which the user can select the dataset to open.
    
    ------------------------    
    output:
    ------------------------
    returns a dictionary containing information regarding the dataset

    current keys are:
      'clock rate' : float64 : the effective digitization rate of the dataset
      'data type'  : str     : the data type as supplied by the user when the data was taken
      'data'       : uint    : the values of the dataset where the bit depth is dependent on the digitizer used.
                               Note that I will not be implementing the code here to convert to appropriate floating point
      'digitizer'  : str     : an identifier for the particular digitizer used, i.e. 3305 for the FPGA or 3302 for the standard digitizer.
                               This is left in to ensure an ID gets carried with the dataset to indicate any appropriate conversions

    ------------------------
    use :
    ------------------------
    Say I want to retrieve a dataset from 'testFile.hdf5'. This file has two datasets, 'ds1' and 'ds2'.
    Then I can extract dataset 'ds1' with the following short code:

    result = openHDF5_dataset('testFile.hdf5')

    and selects its corresponding option or

    result = openHDF5_dataset('testFile.hdf5','ds1')

    which doesn't print anything to screen. This clearly only works if in fact 'ds1' is the FULL NAME
    of a dataset within the HDF5 file.

    REMEMBER TO CLOSE THE FILE WHEN YOU ARE DONE!

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 7/21/2016 - Jeffrey Bonde
      NOTES: Eh, it seems to work.

    ------------------------
    to-do list :
    ------------------------ 
  """

  datasetMap = get_ListOfDatasets(file) # datasets that the program can find
  if len(dataset_name) == 0 :
    dsKeys = list(datasetMap.keys())
    # print out the list of dataset found
    print("\nchoice|      data type      |       size      |  Clock Rate | dataset name")
    print("--------------------------------------------------------------------------")
    for it in range(len(dsKeys)) :
      dataset_dict = datasetMap[dsKeys[it]]
      print('  {0:2d}  : {1:20s}: ({2!s:6}, {3!s:6}): {4:7.3f} MHz : {5}'.format(it, dataset_dict['data type'].decode('UTF8'),*dataset_dict['size'],dataset_dict['clock rate']/10.0**6,dsKeys[it]))

    # get input from user as to which data to open with the option to cancel
    choice = str(eval(input("\n Choose a dataset from the above options to open. Type anything else to cancel.\n"))) # there's uncertain behavior if the user accidentally passes non-empty variable. An error occurs if an empty variable name is passed...hmmm.
    if choice.isdigit() :
      if int(choice) in range(len(dsKeys)) :
        # run *this* function with the chosen dataset
        return openHDF5_dataset(file,dsKeys[int(choice)])
      else :
        print("Cancelling...returning -1")
        return -1
    else :
      print("Cancelling...returning -1")
      return -1
  else : 
    # look for dataset_name in the list of found datasets
    if dataset_name in datasetMap :
      output = {}
    
      # grab the metadata of the retrieved dataset
      output['clock rate'] = datasetMap[dataset_name]['clock rate']
      output['data type'] = datasetMap[dataset_name]['data type']
      output['digitizer'] = datasetMap[dataset_name]['digitizer']
      output['data'] = datasetMap[dataset_name]['dataset_id'] # return the dataset id so that the data can be opened later with a hyperslab
      return output
    else : 
      print("named dataset not found...retrieving available datasets")
      return openHDF5_dataset(file) # re-run as if no datasets are passed

def openHDF5_dataset_automatic(file, index, dataset_name='') :
  """
    extracts a dataset from an HDF5 file

    ------------------------
    inputs:
    ------------------------ 
    file : str : the filename of the HDF5 file from which to extract the dataset
    index: integer: the dataset desired
    dataset_name : str : (optional) if given, searches for the named dataset and extracts it without user input
                         if not given, searches for all datasets in the digitizer group and gives a list
                         from which the user can select the dataset to open.
    
    ------------------------    
    output:
    ------------------------
    returns a dictionary containing information regarding the dataset

    current keys are:
      'clock rate' : float64 : the effective digitization rate of the dataset
      'data type'  : str     : the data type as supplied by the user when the data was taken
      'data'       : uint    : the values of the dataset where the bit depth is dependent on the digitizer used.
                               Note that I will not be implementing the code here to convert to appropriate floating point
      'digitizer'  : str     : an identifier for the particular digitizer used, i.e. 3305 for the FPGA or 3302 for the standard digitizer.
                               This is left in to ensure an ID gets carried with the dataset to indicate any appropriate conversions

    ------------------------
    use :
    ------------------------
    Say I want to retrieve a dataset from 'testFile.hdf5'. This file has two datasets, 'ds1' and 'ds2'.
    Then I can extract dataset 'ds1' with the following short code:

    result = openHDF5_dataset('testFile.hdf5')

    and selects its corresponding option or

    result = openHDF5_dataset('testFile.hdf5','ds1')

    which doesn't print anything to screen. This clearly only works if in fact 'ds1' is the FULL NAME
    of a dataset within the HDF5 file.

    REMEMBER TO CLOSE THE FILE WHEN YOU ARE DONE!

    ------------------------
    edit/status log :
    ------------------------ 
    originally written: 7/21/2016 - Jeffrey Bonde
      NOTES: Eh, it seems to work.

    ------------------------
    to-do list :
    ------------------------ 
  """

  datasetMap = get_ListOfDatasets(file) # datasets that the program can find
  if len(dataset_name) == 0 :
    dsKeys = list(datasetMap.keys())
    if index in range(len(dsKeys)):
      return openHDF5_dataset_automatic(file, index, dsKeys[index])
    else :
      print("Cancelling...returning -1")
      return -1
  else : 
    # look for dataset_name in the list of found datasets
    if dataset_name in datasetMap :
      output = {}
    
      # grab the metadata of the retrieved dataset
      output['clock rate'] = datasetMap[dataset_name]['clock rate']
      output['data type'] = datasetMap[dataset_name]['data type']
      output['digitizer'] = datasetMap[dataset_name]['digitizer']
      output['data'] = datasetMap[dataset_name]['dataset_id'] # return the dataset id so that the data can be opened later with a hyperslab
      return output

def openHDF5(file_name) : # Entirely superfluous. I am still getting used to the "module.function"
  return h5py.File(file_name)

if __name__ == "__main__" :
  """
  # Testing opening datasets
  #fname = '/data_old/BAPSF_Data/Rotation/turbFlowBeta_sept15/053_radLine_diag_trip_bdot_bdotRef_reyn_150v_presScan_380 2015-09-28 14.31.59.hdf5'
  #fname = '/data_old/BAPSF_Data/ICRF_Campaign/April2016/run46_emiss_p35_blockinglimiters_0degreetilt_plane2.hdf5'
  #fname = '/data_old/BAPSF_Data/Axial_Plasma_Jets/AJ8/EmProbe_p30_xyplane_Ar.hdf5'
  fname = '/data_old/BAPSF_Data/Axial_Plasma_Jets/AJ8/EmProbe_p30_xzplane_Ar_bubble_realigned.hdf5'
  file = openHDF5(fname)

  dataset_name='Vp_Diode_7 [Slot 13: SIS 3305 FPGA 1 ch 1]'
  a = openHDF5_dataset(file,dataset_name)

  # gets list of motionlists. If only one motion list is in the HDF5, b[0] is the motion list dictionary
  b = get_MotionLists(file) 

  # converts DAQ dataset dimensions to motion list dimensions
  c = expandDims(a['data'].shape,b[0]) 

  # extracts a hyperslab of 10 points in the first spatial dimension and 1000 points in time
  e = DAQconvert(extractData(a['data'],c,'[2:12,:,:,0:1000]'),a['digitizer'])

  plt.figure(2)
  plt.plot(e[0,0,0,:],'b')
  plt.show()

#  # generate CH plane
#  print "Generating CH-plane"
#  Nx = e.shape[0]
#  Nshots = e.shape[1]
#  Nt = e.shape[2]
#  CH.analyze_dataset(np.reshape(e,(Nx*Nshots,Nt)),5)

  file.close()
  """


  """
  # Testing plotting of the MSI data
  fname = '/data_old/BAPSF_Data/Axial_Plasma_Jets/AJ8/EmProbe_p30_xyplane_Ar.hdf5'
  file = openHDF5(fname)
  a = get_MSI(file,'Discharge')
  print a.keys()

  b = get_MSI(file,'Gas pressure')
  print b.keys()

  c = get_MSI(file,'Interferometer array')
  print c.keys()
 
  d = get_MSI(file,'Magnetic field')
  print d.keys()
 
  e = get_MSI(file)

  file.close()
  """

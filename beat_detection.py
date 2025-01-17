import wave, array, math, time, argparse, sys
import numpy, pywt
from scipy import signal
import pdb
import matplotlib.pyplot as plt

def read_wav(filename):

    #open file, get metadata for audio
    try:
        wf = wave.open(filename,'rb')
    except IOError, e:
        print e
        return

    # typ = choose_type( wf.getsampwidth() ) #TODO: implement choose_type
    nsamps = wf.getnframes();
    assert(nsamps > 0);

    fs = wf.getframerate()
    assert(fs > 0)

    # read entire file and make into an array
    samps = list(array.array('i',wf.readframes(nsamps)))
    #print 'Read', nsamps,'samples from', filename
    try:
        assert(nsamps == len(samps))
    except AssertionError, e:
        print  nsamps, "not equal to", len(samps)
    
    return samps, fs
    
# simple peak detection
def peak_detect(data):
    max_val = numpy.amax(abs(data)) 
    peak_ndx = numpy.where(data==max_val)
    if len(peak_ndx[0]) == 0: #if nothing found then the max must be negative
        peak_ndx = numpy.where(data==-max_val)
    return peak_ndx
    
def bpm_detector(data,fs):
    cA = [] 
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2**(levels-1);
    min_ndx = 60./ 220 * (fs/max_decimation)
    max_ndx = 60./ 40 * (fs/max_decimation)
    
    for loop in range(0,levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA,cD] = pywt.dwt(data,'db4');
            cD_minlen = len(cD)/max_decimation+1;
            cD_sum = numpy.zeros(cD_minlen);
        else:
            [cA,cD] = pywt.dwt(cA,'db4');
        # 2) Filter
        cD = signal.lfilter([0.01],[1 -0.99],cD);

        # 4) Subtractargs.filename out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[::(2**(levels-loop-1))]);
        
        cD = cD - numpy.mean(cD);
        # 6) Recombine the signal before ACF
        #    essentially, each level I concatenate 
        #    the detail coefs (i.e. the HPF values)
        #    to the beginning of the array
        cD_sum = cD[0:cD_minlen] + cD_sum;

    # adding in the approximate data as well...    
    cA = signal.lfilter([0.01],[1 -0.99],cA);
    cA = abs(cA);
    cA = cA - numpy.mean(cA);
    cD_sum = cA[0:cD_minlen] + cD_sum;
    
    # ACF
    correl = numpy.correlate(cD_sum,cD_sum,'full') 
    
    midpoint = len(correl) / 2
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx]);
    peak_ndx_adjusted = peak_ndx[0]+min_ndx;
    bpm = 60./ peak_ndx_adjusted * (fs/max_decimation)
    return bpm,correl
    

def beat_fromfile(filename,window):
    samps,fs = read_wav(filename)
    data = []
    correl=[]
    bpm = 0
    nsamps = len(samps)
    window_samps = int(window*fs)         
    window_ndx = int(1); #current window we are processing
    samps_ndx = 0;  #first sample in window_ndx 
    max_window_ndx = nsamps / window_samps;
    bpms = numpy.zeros(max_window_ndx)
	
    #iterate through all windows
    while window_ndx < max_window_ndx:
        #get a new set of samples
        data = samps[samps_ndx:samps_ndx+window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError( str(len(data) ) ) 
        
        bpms[window_ndx],correl = bpm_detector(data,fs)
        
        #iterate at the end of the loop
        window_ndx = window_ndx + 1;
        samps_ndx = samps_ndx+window_samps;
        
    bpm = numpy.median(bpms)
    return bpm

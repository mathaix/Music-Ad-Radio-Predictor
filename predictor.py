#!/usr/bin/env python

import matplotlib
import pyaudio
import wave
import numpy
import pylab
import time
from math import sqrt
import threading
import datetime
import wave as wv
from pymir import AudioFile

matplotlib.use('TkAgg')
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
_counter = 0
size = 100

from numpy import arange, sin, cos , pi,tan
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import thread

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Radio Ad/Music classifier")

f_fft = Figure(figsize=(10,5), dpi=100)
f_fft.set_facecolor('black')
plot_fft = f_fft.add_subplot(111)
plot_fft.set_axis_bgcolor('black')
canvas_fft = FigureCanvasTkAgg(f_fft, master=root)
canvas_fft.show()
canvas_fft.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#array holding averages
freq_avg = [] #* 100
_counter = 0

#toolbar = NavigationToolbar2TkAgg( canvas, root )
#toolbar.update()
#canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def on_key_event(event):
    print('you pressed %s'%event.key)
    key_press_handler(event, canvas, toolbar)

def _listen():
    root.after(1,listen_loop)

def _write_music():
	_write_file("Music")
	
def _write_Advert():
    _write_file("Advert")
		
def listen_loop():
        import random
	plot_fft.clear()
        canvas_fft.show()
        #t = numpy.arange(0.0, 1.0, 0.01)
        X = numpy.linspace(-2, 2, 256, endpoint=True)
        plot_fft.set_color_cycle(['c', 'm', 'y', 'k'])
	plot_fft.plot(X, numpy.exp(-X)*cos(pi*X*random.randint(-8,9))  + numpy.exp(X)*sin(random.randint(-20,20)*3*pi*X))
	canvas_fft.show()
	
	root.after(1,listen_loop)

def _analyze_file():
    thread.start_new_thread( _analyze_thread , ('analyze-thread',))


def _analyze_thread(thread_name):
    plot_fft.set_axis_bgcolor('black')
    _write_file("analyze")
    _analyze()

def getSpectroidData(audiofile):
    wavData = AudioFile.open(audiofile)
    flattness = float(wavData.spectrum().flatness())
    kurtois = wavData.spectrum().kurtosis()
    centroid = wavData.spectrum().centroid()
    fixedFrames = wavData.frames(1024)
    spectra = [f.spectrum() for f in fixedFrames]
    return (flattness, kurtois, centroid, spectra)

def getCentroidData(spectra):
    centroid_data = [spectra[i].centroid() for i in range(0,len(spectra))]
    return (numpy.max(centroid_data),numpy.min(centroid_data),
            numpy.mean(centroid_data),
            numpy.std(centroid_data))

def _analyze():
    fname = "Tmp/tmp.wav"
    (sample_rate, input_signal) = get_wav_info(fname)
    (rfreqs, fft_mag) = compute_fft(float(sample_rate), input_signal)
    (flattness, kurtois, centroid, spectra) = getSpectroidData(fname)
    (high, low, mean, std) = getCentroidData(spectra)
    result1 = model(numpy.mean(fft_mag[0:2000]), numpy.mean(fft_mag[2000:4000]), numpy.mean(fft_mag), high, low, mean, std)
    result2 = model_2(numpy.mean(fft_mag[0:2000]), high, centroid)
    print "probablilites {}".format(result1, result2)
    if result1 > 0.6 :
        plot_fft.set_axis_bgcolor('blue')
    else:
        plot_fft.set_axis_bgcolor('red')
        

def get_wav_info(wav_file):
    wav = wv.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return frame_rate,sound_info

def compute_fft(sample_rate,input_signal):
    fft_out = numpy.fft.rfft(input_signal)
    fft_mag = [sqrt(i.real**2 + i.imag**2)/len(fft_out) for i in fft_out]
    num_samples = len(input_signal)
    rfreqs = [(i*1.0/num_samples)*sample_rate for i in range(num_samples//2+1)]   
    return (rfreqs, fft_mag)

def model(low, high, mean, centroid_high, centroid_low, centroid_mean, centroid_std):
    print low, high, mean,centroid_high, centroid_low, centroid_mean, centroid_std
    result = 8.028 - 0.187*low + 2.157*mean - 1.118e-3*centroid_high -8.609e-5*centroid_low + 3.684e-4*centroid_mean -2.216e-3*centroid_std
    return logistic(result)
 
def model_2 (low, centroid_high, centroid):
    result = 4.047 - 0.0834644*low - 0.0017282*centroid_high + 0.0026275*centroid
    return logistic(result)

def logistic(z):
    return 1.0 / (1.0 + numpy.exp(-z))

def _write_file(filetype):
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 3
	
	if filetype=="Advert":
		WAVE_OUTPUT_FILENAME = "Advert/output%s.wav" % datetime.datetime.now().strftime("%y%m%d%H%M%S")
	elif filetype=="Music":
		WAVE_OUTPUT_FILENAME = "Music/output%s.wav" % datetime.datetime.now().strftime("%y%m%d%H%M%S")
        else:
            WAVE_OUTPUT_FILENAME = "Tmp/tmp.wav"
	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

	print("* recording")

	frames = []
	
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
    	
	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.RIGHT)
button = Tk.Button(master=root, text='Analyze', command=_analyze_file)
button.pack(side=Tk.LEFT)
_listen()


Tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.
#0201504700

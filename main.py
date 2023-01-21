
def GetFile():
    import scipy.io as spio
    wav_fname = "D:\\Git\\Voyager_Golden_Record\\mono_left.wav"
    #spio.wavfile.read(wav_fname)
    samplerate, data = spio.wavfile.read(wav_fname)
    length = data.shape[0] / samplerate
    print(f"length = {length}s")
    time = np.linspace(0., length, data.shape[0])
    #plt.plot(time, data[:], label="Left channel")
    #plt.xlabel("Time [s]")
    #plt.ylabel("Amplitude")
    #plt.show()
    return samplerate, data
 
from pydub import AudioSegment
from pydub.utils import get_array_type
import os
import array
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from PIL import Image
import time
# sound = AudioSegment.from_mp3('voyager.mp3')
# left = sound.split_to_mono()[0]
# bit_depth = left.sample_width * 8
# array_type = get_array_type(bit_depth)
# numeric_array = array.array(array_type, left._data)
# from pydub.utils import mediainfo
# info = mediainfo('voyager.mp3')
# print(info['sample_rate'])
# samplerate = int(info['sample_rate'])
# print(samplerate)
# data = numeric_array

samplerate, data = GetFile()
print("try 5 windows with for and global adjust")
from scipy.signal import butter, filtfilt
import pywt
cA, cD = pywt.dwt(data,'haar','per')
cD = cD*0
y = pywt.idwt(cA,cD,'haar','per')

image_data = []
window_size = 735
line_hz = samplerate / (window_size)
nyquist = samplerate / 2
Wn = line_hz / nyquist
#window = np.zeros([1,window_size])

offset = int(samplerate*31.25)+300 #start time
print(f" offset:{offset}" )
buufer = -200  
adjust = 0
for SignalTime in range(0,900,9):
    offset = int(samplerate*(SignalTime + 31.25))+300 #start time
    image_data = []
    for index in range(735):
        b, a = butter(3, Wn, 'highpass',analog='true')
        adjust = int(index *-1.05)

        window = y[offset + buufer + adjust :offset+window_size + buufer + adjust]
        image_data.append(-1*window[0:len(window)])

        offset = offset + window_size
        # plt.plot(range(len(window)),window[:])
        # plt.show()


    image_data=np.stack(image_data) 
    image_data=255/2* (image_data/image_data.max()) + 255/2
    image_data=image_data.astype('uint8')  

    plt.imshow(image_data,cmap="gray")
    plt.ion()
    plt.show()
    plt.pause(1)
    plt.ioff()
    plt.clf()




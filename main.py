
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

sound = AudioSegment.from_mp3('voyager.mp3')
left = sound.split_to_mono()[0]
bit_depth = left.sample_width * 8
array_type = get_array_type(bit_depth)
numeric_array = array.array(array_type, left._data)
from pydub.utils import mediainfo
info = mediainfo('voyager.mp3')
print(info['sample_rate'])
samplerate = int(info['sample_rate'])
print(samplerate)

data = numeric_array
#samplerate, data = GetFile()
print("try 5 windows with for and global adjust")
from scipy.signal import butter, filtfilt
image_data = []
window_size = 734
line_hz = samplerate / (3*window_size)
nyquist = samplerate / 2
Wn = line_hz / nyquist
#window = np.zeros([1,window_size])

offset = int(samplerate*31.25) #start time
print(f" offset:{offset}" )
buufer = 50  

for index in range(180):
    b, a = butter(3, Wn, 'highpass',analog='true')
    #adjust = int(index *window_size/600)
    adjust = int((1.0*offset - samplerate*31.25)/samplerate *window_size/30) 

    for j in range(5):
        window = data[offset+j*window_size+buufer + adjust :offset+(j+1)*window_size+buufer + adjust]
        #print(len(window))
        x = filtfilt(b, a, window)
        w = np.clip(x,-2500,2500)
        image_data.append(-1*w[0:len(w)])
        #plt.plot(range(len(w)), w[:])

    offset = offset + int(3*window_size) 
    #print(f"new offset:{offset}" )
    #print(f"end at{offset+(j+1)*window_size+buufer + adjust}") 
plt.imshow(image_data,cmap="gray")
plt.show()
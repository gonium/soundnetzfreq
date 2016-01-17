import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
#for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
for i in range (0,numdevices):
  if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
    print "Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name')

  if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
    print "Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name')

devinfo = p.get_device_info_by_index(4)
print "Selected device is ", devinfo.get('name')
print devinfo
if p.is_format_supported(9600.0,  # Sample rate
                         input_device=devinfo["index"],
                         input_channels=devinfo['maxInputChannels'],
                         input_format=pyaudio.paInt16):
  print 'Yay!'


#####################################


CHUNK = 512
WIDTH = 2
CHANNELS = 2
RATE = 9600

def callback(in_data, frame_count, time_info, status):
  samples = np.fromstring(in_data, dtype=np.int16)
  samples = np.reshape(samples, (CHUNK, WIDTH))
  left = samples[:, 0]
  right = samples[:, 1]
  print "%d - %s" % (len(right), np.array_str(right))
  #return (in_data, pyaudio.paContinue)

  plt.figure(1, figsize=(15,9))
  plt.clf()
  plt.plot(np.arange(len(right)), right)
  plt.savefig("sample-waveform.pdf")
  return (in_data, pyaudio.paComplete)



stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                input_device_index=devinfo["index"],
                rate=RATE,
                input=True,
                stream_callback=callback,
                frames_per_buffer=CHUNK)

print("* recording")

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

print("* done")

stream.stop_stream()
stream.close()

p.terminate()

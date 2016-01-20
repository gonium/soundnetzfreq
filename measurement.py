import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import time
import argparse
import sys

CHUNK = 8*512    # 512 16bit values
CHANNELS = 1   # mono
RATE = 44100

class Recorder(object):
  def __enter__(self):
    self.p = pyaudio.PyAudio()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.p.terminate()

  def print_devicelist(self):
    info = self.p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
      if self.p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
          print "Input Device id ", i, " - ", self.p.get_device_info_by_host_api_device_index(0,i).get('name')

  def check_format(self, device_index):
    devinfo = self.p.get_device_info_by_index(device_index)
    print "Selected device is ", devinfo.get('name')
    #print devinfo
    if self.p.is_format_supported(RATE,  # Sample rate
                             input_device=devinfo["index"],
                             input_channels=devinfo['maxInputChannels'],
                             input_format=pyaudio.paFloat32):
      print 'Sound device supports default mode'
    else:
      raise ValueError('Sound device does not support default mode')

  def start(self, device_index):
    def recording_callback(in_data, frame_count, time_info, status):
        samples = np.fromstring(in_data, dtype=np.float32)
        print "%d - %s" % (len(samples), np.array_str(samples))
        # TODO: Forward the samples into the queue and request next one
        #return (in_data, pyaudio.paContinue)
        # quick and dirty debugging: plot samples
        plt.figure(1, figsize=(15,9))
        plt.clf()
        plt.plot(np.arange(len(samples)), samples)
        plt.savefig("sample-waveform.png")
        return (in_data, pyaudio.paComplete)
    self.stream = self.p.open(
              format = pyaudio.paFloat32,
              channels=CHANNELS,
              input_device_index=device_index,
              rate=RATE,
              input=True,
              stream_callback=recording_callback,
              frames_per_buffer=CHUNK)
    self.stream.start_stream()
    while self.stream.is_active():
        time.sleep(0.1)
    self.stream.stop_stream()
    self.stream.close()

if __name__ == "__main__":
  cmd_parser = argparse.ArgumentParser()
  cmd_parser.add_argument("device", help="sound device id to use for \
      sampling", nargs='?')
  cmd_parser.add_argument("--list", help="prints a list of audio devices \
    on this system", action='store_true', default=False)
  args = cmd_parser.parse_args()

  with Recorder() as rec:
    if args.list:
      print "Searching for audio input devices"
      rec.print_devicelist()
      sys.exit(1)
    if args.device == None:
      print "Please provide the ID of the input device"
      sys.exit(2)
    else:
      device_id=int(args.device)
      print "Using input device %d" % device_id
      rec.check_format(device_id)
      rec.start(device_id)


print "Importing libraries"
import scipy.signal as ss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#------------------------------------------------
# 0. Global definitions
#------------------------------------------------
# Use sample rate of 9,6 kHz, "record" for one second
sample_rate = 9600.0
nsamples = 9600

#------------------------------------------------
# 1. Create noisy input signal
#------------------------------------------------
def create_signal(F_base, static):
  A_base = 2.0
  F_noise1 = 500.0
  A_noise1 = 0.1
  F_noise2 = 1500.0
  A_noise2 = 0.1
  A_static = 0.1

  t = np.arange(nsamples) / sample_rate
  truth = A_base * np.sin(2*np.pi*F_base*t)
  signal = truth + \
    A_noise1*np.sin(2*np.pi*F_noise1*t) + \
    A_noise2*np.sin(2*np.pi*F_noise2*t) + \
    A_static * static
  signal = signal - np.median(signal)
  return (t, truth, signal)

#------------------------------------------------
# 2. Create a FIR filter and apply it to signal.
#------------------------------------------------
def calc_fir_coeff(cutoff_hz, numtabs, sample_rate):
  # The Nyquist rate of the signal.
  nyq_rate = sample_rate / 2.
  # The cutoff frequency of the filter
  cutoff_hz = 52.0
  # Length of the filter (number of coefficients, i.e. the filter order + 1)
  # Use firwin to create a lowpass FIR filter
  # Note: sample rate, cutoff frequency etc. all relate. Needs adequate
  # choice of parameters.
  fir_coeff = ss.firwin(numtaps, cutoff_hz/nyq_rate)
  #print "Cutoff: %.3f Hz, Nyqist: %.3f Hz" % (cutoff_hz, nyq_rate)
  return fir_coeff

def fir_filter(signal, fir_coeff):
  filtered_signal = ss.lfilter(fir_coeff, 1.0, signal)
  return filtered_signal

#------------------------------------------------
# Calculate frequency based on zero crossings
#------------------------------------------------
def calc_frequency(signal):
  zero_crossing = 0.0
  # Find all indices right before a rising-edge zero crossing
  indices = np.where((signal[1:] >= zero_crossing) & \
      (signal[:-1] < zero_crossing))
  # More accurate, using linear interpolation to find intersample
  # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
  crossings = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]
  num_zerocrossing_samples = np.median(np.diff(crossings))
  freq = (sample_rate/num_zerocrossing_samples)
  #print "Mean number of samples between zero crossings: %.3f" % num_zerocrossing_samples
  return freq


#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
def plot_signalquality(t, truth, signal, filtered, freq):
  # The first N-1 samples are "corrupted" by the initial conditions
  warmup = numtaps - 1
  # The phase delay of the filtered signal
  delay = (warmup / 2) / sample_rate
  # The delay is based on the time t. We also need the number of 
  # delay samples in order to calculate things.
  sample_shift = warmup/2
  # now, do the plotting.
  plt.figure(1, figsize=(12,9))
  plt.clf()
  plt.subplots_adjust(hspace=.7)
  plt.subplot(4,1,1)
  plt.title("Vergleich Wahrheit/Rauschen(%.3f Hz)" % freq)
  plt.plot(t, truth, 'g-', label="Wahrheit")
  plt.plot(t, signal, 'b-', label="Input")
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(4,1,2)
  plt.title("Vergleich vor/nach FIR-Filter (%.3f Hz)" % freq)
  plt.plot(t, signal, 'b-', label="Input")
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t-delay, filtered, 'r-', label="Filtered", linewidth=2)
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(4,1,3)
  plt.title("Vergleich Wahrheit/FIR-Filter (%.3f Hz)" % freq)
  plt.plot(t, truth, 'g-', label="Wahrheit")
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t-delay, filtered, 'r-', label="Filtered", linewidth=1)
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(4,1,4)
  plt.title("Abweichung Wahrheit/FIR-Filter (%.3f Hz)" % freq)
  deviation = []
  for i in range(int(truth.size-sample_shift)):
    deviation.append(truth[i] - filtered[i+sample_shift])
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t[:-sample_shift], deviation, 'k-', label="Abweichung", linewidth=2)
  plt.xlim([0.0, 0.06])
  plt.savefig("images/signalquality-%.3f.png" % freq)


#------------------------------------------------
# Main loop.
#------------------------------------------------
print "Starting calculations"
# create static around 0.0
static = np.random.random_sample(nsamples)-0.5
frequencies = np.arange(49.7, 50.3, 0.05)
targetfrequencies = []
measuredfrequencies = []
numtaps = 39
cutoff_freq_hz = 52.0
fir_coeff = calc_fir_coeff(cutoff_freq_hz, numtaps, sample_rate)
for idx, target in enumerate(frequencies):
  t, truth, signal = create_signal(target, static)
  filtered = fir_filter(signal, fir_coeff)
  freq = calc_frequency(filtered)
  targetfrequencies.append(target)
  measuredfrequencies.append(freq)
  print "Deviation: %.2f mHz, signal mean: %.5f - target frequency: %.3f, measured frequency: %.3f" % ((target - freq)*1000, np.mean(signal), target, freq)
  plot_signalquality(t, truth, signal, filtered, target)

#------------------------------------------------
# plot overall stats.
#------------------------------------------------
xlims = (np.min(targetfrequencies), np.max(targetfrequencies))
ylims = (np.min(measuredfrequencies), np.max(measuredfrequencies))
plt.figure(2)
plt.clf()
plt.plot(targetfrequencies, measuredfrequencies, 'k.')
plt.title("Frequenz: Wahrheit vs. Messung")
plt.xlabel("Wahre Frequenz [Hz]")
plt.ylabel("Gemessene Frequenz[Hz]")
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig("images/target_vs_measured.png")

plt.figure(2)
plt.clf()
delta = np.asarray(measuredfrequencies) - np.asarray(targetfrequencies)
plt.plot(targetfrequencies, (delta), 'k.')
plt.title("Abweichung (Sampling %d Hz, FIR w/ %d taps)" % (sample_rate,
  numtaps))
plt.xlabel("Wahre Frequenz [Hz]")
plt.ylabel("Abweichung gemessene Frequenz [Hz]")
plt.xlim(xlims)
plt.savefig("images/measurement_inaccurency.png")


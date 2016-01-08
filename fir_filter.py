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
numtaps = 51

#------------------------------------------------
# 1. Create noisy input signal
#------------------------------------------------
def create_signal(F_base):
  A_base = 1.0
  F_noise1 = 500.0
  A_noise1 = 0.25
  F_noise2 = 1500.0
  A_noise2 = 0.25
 # A_rand = 0.01

  noise_rand = np.arange(-1,1,nsamples)
  t = np.arange(nsamples) / sample_rate
  signal = A_base * np.sin(2*np.pi*F_base*t) + \
    A_noise1*np.sin(2*np.pi*F_noise1*t) + \
    A_noise2*np.sin(2*np.pi*F_noise2*t)# + \
    #A_rand * noise_rand
  return (t, signal)

#------------------------------------------------
# 2. Create a FIR filter and apply it to signal.
#------------------------------------------------
def fir_filter(signal):
  # The Nyquist rate of the signal.
  nyq_rate = sample_rate / 2.
  # The cutoff frequency of the filter
  cutoff_hz = 55.0
  # Length of the filter (number of coefficients, i.e. the filter order + 1)
  # Use firwin to create a lowpass FIR filter
  # Note: sample rate, cutoff frequency etc. all relate. Needs adequate
  # choice of parameters.
  fir_coeff = ss.firwin(numtaps, cutoff_hz/nyq_rate)
  print "Cutoff: %.3f Hz, Nyqist: %.3f Hz" % (cutoff_hz, nyq_rate)
  filtered_signal = ss.lfilter(fir_coeff, 1.0, signal)
  return filtered_signal

#------------------------------------------------
# Calculate frequency based on zero crossings
#------------------------------------------------
def calc_frequency(signal):
  # Find all indices right before a rising-edge zero crossing
  indices = np.where((signal[1:] >= 0) & (signal[:-1] < 0))
  # More accurate, using linear interpolation to find intersample
  # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
  crossings = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]
  num_zerocrossing_samples = np.mean(np.diff(crossings))
  freq = (sample_rate/num_zerocrossing_samples)
  print "Mean number of samples between zero crossings: %.3f" % num_zerocrossing_samples
  return freq


#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
def plot_signalquality(t, signal, filtered, freq):
  # The first N-1 samples are "corrupted" by the initial conditions
  warmup = numtaps - 1
  # The phase delay of the filtered signal
  delay = (warmup / 2) / sample_rate
  plt.figure(1)
  plt.clf()
  plt.plot(t, signal, 'b-', label="Input")
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t-delay, filtered, 'r-', label="Filtered", linewidth=2)
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.title("Vergleich mit/ohne Filter (%.3f Hz)" % freq)
  plt.savefig("images/signalquality-%.3f.png" % freq)


#------------------------------------------------
# Main loop.
#------------------------------------------------
frequencies = np.arange(49.7, 50.3, 0.001)
targetfrequencies = []
measuredfrequencies = []
for idx, target in enumerate(frequencies):
  t, signal = create_signal(target)
  filtered = fir_filter(signal)
  freq = calc_frequency(filtered)
  targetfrequencies.append(target)
  measuredfrequencies.append(freq)
  print "Target frequency: %.3f, measured frequency: %.3f" % (target,
      freq)
  #plot_signalquality(t, signal, filtered, target)

#------------------------------------------------
# plot overall stats.
#------------------------------------------------
xlims = (np.min(targetfrequencies), np.max(targetfrequencies))
ylims = (np.min(measuredfrequencies), np.max(measuredfrequencies))
plt.figure(2)
plt.clf()
plt.plot(targetfrequencies, measuredfrequencies, 'k.')
plt.title("Target vs. measured frequencies")
plt.xlabel("Target frequency [Hz]")
plt.ylabel("Measured frequency [Hz]")
plt.xlim(xlims)
plt.ylim(ylims)
plt.savefig("images/target_vs_measured.png")

plt.figure(2)
plt.clf()
delta = np.asarray(measuredfrequencies) - np.asarray(targetfrequencies)
plt.plot(targetfrequencies, (delta), 'k.')
plt.title("Inaccurency (sampling at %d Hz, FIR w/ %d taps)" % (sample_rate,
  numtaps))
plt.xlabel("Target frequency [Hz]")
plt.ylabel("Delta frequency [Hz]")
plt.xlim(xlims)
plt.savefig("images/measurement_inaccurency.png")


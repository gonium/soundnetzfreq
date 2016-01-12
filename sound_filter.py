print "Importing libraries"
import scipy.signal as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
def create_signal(F_base, static, snr):
  A_base = 1.0 * snr
  F_noise1 = 500.0
  A_noise1 = 0.33
  F_noise2 = 1500.0
  A_noise2 = 0.33
  A_static = 0.34

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
def calc_freq_zerocrossing(signal):
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
# From https://github.com/endolith/waveform-analyzer/blob/master/common.py
# Quadratic interpolation for estimating the true position of an
# inter-sample maximum when nearby samples are known.
#------------------------------------------------
def parabolic(f, x):
  """Quadratic interpolation for estimating the true position of an
  inter-sample maximum when nearby samples are known.
  f is a vector and x is an index for that vector.
  Returns (vx, vy), the coordinates of the vertex of a parabola that goes
  through point x and its two neighbors.
  Example:
  Defining a vector f with a local maximum at index 3 (= 6), find local
  maximum if points 2, 3, and 4 actually defined a parabola.
  In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
  In [4]: parabolic(f, argmax(f))
  Out[4]: (3.2142857142857144, 6.1607142857142856)
  """
  xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
  yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
  return (xv, yv)

#------------------------------------------------
# Calculate frequency based on fft - see
# https://github.com/endolith/waveform-analyzer/blob/master/frequency_estimator.py
#------------------------------------------------
def calc_freq_fft(signal):
  N = len(signal)
  # Compute Fourier transform of windowed signal
  windowed = signal * ss.kaiser(N, 100)
  f = np.fft.rfft(windowed)
  # Find the peak and interpolate to get a more accurate peak
  i_peak = np.argmax(abs(f)) # Just use this value for less-accurate result
  i_interp = parabolic(np.log(abs(f)), i_peak)[0]
  # Convert to equivalent frequency
  return sample_rate * i_interp / N # Hz

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
def plot_signalquality(t, truth, signal, filtered, snr, freq):
  # The first N-1 samples are "corrupted" by the initial conditions
  warmup = numtaps - 1
  # The phase delay of the filtered signal
  delay = (warmup / 2) / sample_rate
  # The delay is based on the time t. We also need the number of 
  # delay samples in order to calculate things.
  sample_shift = warmup/2
  # now, do the plotting.
  plt.figure(1, figsize=(15,9))
  plt.clf()
  plt.subplots_adjust(hspace=.7)
  plt.subplot(5,1,1)
  plt.title("Vergleich Signal/Rauschen - (%.3f Hz, SNR %.1f)" % (freq,
    snr))
  plt.plot(t, truth, 'g-', label="Signal")
  plt.plot(t, signal, 'b-', label="Signal+Rauschen")
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(5,1,2)
  plt.title("Vergleich vor/nach FIR-Filter")
  plt.plot(t, signal, 'b-', label="Input")
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t-delay, filtered, 'r-', label="Filtered", linewidth=2)
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(5,1,3)
  plt.title("Vergleich Wahrheit/FIR-Filter")
  plt.plot(t, truth, 'g-', label="Wahrheit")
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t-delay, filtered, 'r-', label="Filtered", linewidth=1)
  plt.xlim([0.0, 0.06])
  plt.legend(loc="best")
  plt.subplot(5,1,4)
  plt.title("Abweichung Wahrheit/FIR-Filter")
  deviation = []
  for i in range(int(truth.size-sample_shift)):
    deviation.append(truth[i] - filtered[i+sample_shift])
  # Plot the filtered signal, shifted to compensate for the phase delay
  plt.plot(t[:-sample_shift], deviation, 'k-', label="Abweichung", linewidth=2)
  plt.xlim([0.0, 0.06])
  plt.subplot(5,1,5)
  plt.title("Power spectrum")
  f, Pxx_spec = ss.periodogram(signal, sample_rate, 'flattop', scaling='spectrum')
  plt.semilogy(f, np.sqrt(Pxx_spec))
  plt.xlabel('frequency [Hz]')
  plt.ylabel('Linear spectrum [V RMS]')
  plt.xlim([0, 1600])
  plt.savefig("images/signalquality-%.1f-%.3f.png" % (snr, freq))


#------------------------------------------------
# Main loop.
#------------------------------------------------
print "Starting calculations"
# creates uniform static around 0.0
# static = np.random.random_sample(nsamples)-0.5
# Create gaussian noise around 0.0
static = np.random.normal(0.0, 1.0, nsamples)
frequencies = np.arange(49.7, 50.3, 0.05)
snrs = np.arange(1.0, 3.0, 0.5)
numtaps = 39
cutoff_freq_hz = 52.0
fir_coeff = calc_fir_coeff(cutoff_freq_hz, numtaps, sample_rate)
df = pd.DataFrame()
for idx, snr in enumerate(snrs):
  print "### SNR: %.1f" % snr
  for jdx, target in enumerate(frequencies):
    t, truth, signal = create_signal(target, static, 5.0)
    filtered = fir_filter(signal, fir_coeff)
    #freq = calc_freq_zerocrossing(filtered)
    freq = calc_freq_fft(filtered)
    deviation = (target - freq)*1000
    data = pd.DataFrame({"SNR": snr, "Deviation": deviation, "Signal":
      target, "Measured": freq, "Mean": np.mean(signal)}, index=[0])
    df = df.append(data)
    print "Deviation: %.2f mHz, signal mean: %.5f - target: %.3f, measured: %.3f" % ((target - freq)*1000, np.mean(signal), target, freq)
    #plot_signalquality(t, truth, signal, filtered, snr, target)

print df.describe()

#------------------------------------------------
# plot overall stats.
#------------------------------------------------
xlims = (np.min(df.Signal), np.max(df.Signal))
ylims = (np.min(df.Measured), np.max(df.Measured))

plt.figure(2)
plt.clf()
for idx, snr in enumerate(snrs):
  snr_df = df[df.SNR == snr]
  delta = snr_df.Measured - snr_df.Signal
  plt.plot(snr_df.Signal, delta, marker=".", label="SNR %.1f" % snr)
plt.title("Abweichung (Sampling %d Hz, FIR w/ %d taps)" % (sample_rate,
  numtaps))
plt.xlabel("Wahre Frequenz [Hz]")
plt.ylabel("Abweichung gemessene Frequenz [Hz]")
plt.xlim(xlims)
plt.legend(loc="best")
plt.savefig("images/target_vs_measured-relative.png")


from numpy import sin, arange, pi
from scipy.signal import lfilter, firwin
from pylab import figure, plot, grid, show
 
#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------
# samples of (1000Hz + 15000 Hz) at 48 kHz
sample_rate = 3000.
nsamples = 1000
 
F_base = 50.0
A_base = 1.0
 
F_noise = 500.0
A_noise = 0.25
 
t = arange(nsamples) / sample_rate
signal = A_base * sin(2*pi*F_base*t) + A_noise*sin(2*pi*F_noise*t)
 
#------------------------------------------------
# Create a FIR filter and apply it to signal.
#------------------------------------------------
# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.
 
# The cutoff frequency of the filter
cutoff_hz = 60.0
 
# Length of the filter (number of coefficients, i.e. the filter order + 1)
numtaps = 29
 
# Use firwin to create a lowpass FIR filter
#fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate)
fir_coeff = firwin(numtaps, cutoff_hz/nyq_rate)
# Note: If the sample rate is too high the filter does not work!

print "Cutoff: %.3f Hz, Nyqist: %.3f Hz" % (cutoff_hz, nyq_rate)

# Use lfilter to filter the signal with the FIR filter
filtered_signal = lfilter(fir_coeff, 1.0, signal)
 
#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
 
# The first N-1 samples are "corrupted" by the initial conditions
warmup = numtaps - 1
 
# The phase delay of the filtered signal
delay = (warmup / 2) / sample_rate
 
figure(1)
# Plot the original signal
plot(t, signal)
 
# Plot the filtered signal, shifted to compensate for the phase delay
plot(t-delay, filtered_signal, 'r-')
 
## Plot just the "good" part of the filtered signal.  The first N-1
## samples are "corrupted" by the initial conditions.
#plot(t[warmup:]-delay, filtered_signal[warmup:], 'g', linewidth=4)
 
grid(True)
 
show()
 
#------------------------------------------------
# Print values
#------------------------------------------------
def print_values(label, values):
    var = "float32_t %s[%d]" % (label, len(values))
    print "%-30s = {%s}" % (var, ', '.join(["%+.10f" % x for x in values]))
 
print_values('signal', signal)
print_values('fir_coeff', fir_coeff)
print_values('filtered_signal', filtered_signal)

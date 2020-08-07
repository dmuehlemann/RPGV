import scipy.signal as signal
import matplotlib.pyplot as plt

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.1 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')


temp = z_djf_org[:50,1,1]
# Second, apply the filter
tempf = signal.filtfilt(B,A, temp)

# Make plots
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(temp, 'b-')
plt.plot(tempf, 'r-',linewidth=2)
plt.ylabel("Temperature (oC)")
plt.legend(['Original','Filtered'])
plt.title("Temperature from LOBO (Halifax, Canada)")
ax1.axes.get_xaxis().set_visible(False)

ax1 = fig.add_subplot(212)
plt.plot(temp-tempf, 'b-')
plt.ylabel("Temperature (oC)")
plt.xlabel("Date")
plt.legend(['Residuals'])
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, lfilter
matplotlib.use('TkAgg')

CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1

def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def high_pass_filter(signal, cutoff=100, fs=RATE, order=5):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return lfilter(b, a, signal)

def median_filter(signal, window_size=5):
    result = np.median(np.lib.stride_tricks.sliding_window_view(signal, window_size), axis=-1)
    pad_width = len(signal) - len(result)
    result = np.pad(result, (pad_width // 2, pad_width - pad_width // 2), mode='edge')
    return result

p = pyaudio.PyAudio()

print("마이크 장치 목록:")
input_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        input_devices.append(i)
        print(f"{i}: {info['name']} ({int(info['maxInputChannels'])} 채널)")

device_index = int(input("\n사용할 마이크 장치의 번호를 입력하세요: "))

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

fig, (ax_original, ax_filtered, ax_fft) = plt.subplots(3, 1, figsize=(8, 6))
# fig, (ax_original, ax_filtered) = plt.subplots(2, 1, figsize=(8, 6))
x_time = np.arange(0, CHUNK)

line_original, = ax_original.plot(x_time, np.zeros(CHUNK), label="Original", color='gray')
ax_original.set_ylim(-32768, 32767)
ax_original.set_title("Original Waveform")

line_filtered, = ax_filtered.plot(x_time, np.zeros(CHUNK), label="Filtered", color='blue')
ax_filtered.set_ylim(-32768, 32767)
ax_filtered.set_title("Noise-Filtered Waveform")

x_fft = np.fft.rfftfreq(CHUNK, d=1./RATE)
line_fft, = ax_fft.semilogx(x_fft, np.zeros(len(x_fft)), label="FFT", color='red')
ax_fft.set_xlim(20, RATE/2)
ax_fft.set_ylim(0, 30000)
ax_fft.set_title("FFT Spectrum (Noise Removed)")

max_freq_line = ax_fft.axvline(x=0, color='green', linestyle='--', label='Peak Frequency')

plt.tight_layout()
plt.show(block=False)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)

        filtered = samples.astype(np.float32)
        filtered = high_pass_filter(filtered)
        filtered = median_filter(filtered, window_size=5)

        line_original.set_ydata(samples)
        line_filtered.set_ydata(filtered)

        fft_vals = np.abs(np.fft.rfft(filtered))
        cleaned_fft = np.clip(fft_vals, 0, None)
        line_fft.set_ydata(cleaned_fft)

        peak_index = np.argmax(cleaned_fft)
        peak_freq = x_fft[peak_index]
        max_freq_line.set_xdata([peak_freq, peak_freq])
        ax_fft.set_title(f"FFT Spectrum - Peak: {int(peak_freq)} Hz")

        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping audio stream...")

stream.stop_stream()
stream.close()
p.terminate()

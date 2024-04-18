# ASEHW3
Homework3 Fast Convolution

## Runtime Table
**audio** example: 1sec audio signal convolve with 1sec IR \
**vector** example: 10000 samples signal convolve 1000 samples IR

|          | Time Domain | Freq Domain |
|----------|----------|----------|
| audio (8192 hopsize) | +3min | 1s |
| audio (2048 hopsize) | +12min | 6s |
| vector (512 hopsize) | 0.55s | 0.02s |
| vector (256 hopsize) | 0.91s | 0.14s |
| vector (64 hopsize) | 1.82s | 0.15s |
| vector (4 hopsize) | 23.27s | 5.28s |

From the runtime table, we can see that the frequency domain convolution is much more efficient than the time domain convolution when doing on large sample audio data. Less hopsize will also increase the computation time.


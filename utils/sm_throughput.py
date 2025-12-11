import matplotlib.pyplot as plt
import re
import matplotlib.ticker as ticker

# Raw measurement data
raw_data_reduce = """
Size 16 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.18 GB/s
Size 16384 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.43 GB/s
Size 32768 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.78 GB/s
Size 65536 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.82 GB/s
Size 131072 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.83 GB/s
Size 262144 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 1.65 GB/s
Size 524288 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 3.28 GB/s
Size 1048576 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.50 GB/s
Size 2097152 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.59 GB/s
Size 4194304 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.67 GB/s
Size 8388608 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.71 GB/s
Size 16777216 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.75 GB/s
Size 33554432 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.76 GB/s
Size 67108864 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.50 GB/s
Size 134217728 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.51 GB/s
Size 268435456 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.52 GB/s
Size 536870912 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.52 GB/s
Size 1073741824 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.53 GB/s
Size 2147483648 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.53 GB/s
Size 16 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.10 GB/s
Size 8192 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.19 GB/s
Size 16384 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.41 GB/s
Size 32768 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.79 GB/s
Size 65536 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.82 GB/s
Size 131072 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.83 GB/s
Size 262144 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 1.65 GB/s
Size 524288 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 3.29 GB/s
Size 1048576 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 6.50 GB/s
Size 2097152 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.95 GB/s
Size 4194304 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 13.14 GB/s
Size 8388608 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 13.30 GB/s
Size 16777216 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 13.39 GB/s
Size 33554432 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 13.43 GB/s
Size 67108864 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.80 GB/s
Size 134217728 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.83 GB/s
Size 268435456 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.84 GB/s
Size 536870912 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.86 GB/s
Size 1073741824 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.86 GB/s
Size 2147483648 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.87 GB/s
Size 16 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.20 GB/s
Size 16384 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.36 GB/s
Size 32768 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.79 GB/s
Size 65536 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.82 GB/s
Size 131072 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.83 GB/s
Size 262144 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 1.65 GB/s
Size 524288 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 3.29 GB/s
Size 1048576 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 6.50 GB/s
Size 2097152 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 12.94 GB/s
Size 4194304 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 25.44 GB/s
Size 8388608 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 25.86 GB/s
Size 16777216 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 26.22 GB/s
Size 33554432 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 26.42 GB/s
Size 67108864 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.66 GB/s
Size 134217728 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.73 GB/s
Size 268435456 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.77 GB/s
Size 536870912 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.79 GB/s
Size 1073741824 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.81 GB/s
Size 2147483648 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 24.82 GB/s
Size 16 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.19 GB/s
Size 16384 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.41 GB/s
Size 32768 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.79 GB/s
Size 65536 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.82 GB/s
Size 131072 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.83 GB/s
Size 262144 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 1.65 GB/s
Size 524288 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 3.29 GB/s
Size 1048576 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 6.50 GB/s
Size 2097152 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 12.94 GB/s
Size 4194304 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 25.44 GB/s
Size 8388608 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 47.57 GB/s
Size 16777216 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 48.81 GB/s
Size 33554432 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 49.67 GB/s
Size 67108864 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.32 GB/s
Size 134217728 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.58 GB/s
Size 268435456 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.73 GB/s
Size 536870912 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.84 GB/s
Size 1073741824 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.91 GB/s
Size 2147483648 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 46.94 GB/s
Size 16 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.19 GB/s
Size 16384 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.39 GB/s
Size 32768 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.79 GB/s
Size 65536 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.82 GB/s
Size 131072 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.83 GB/s
Size 262144 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 1.65 GB/s
Size 524288 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 3.28 GB/s
Size 1048576 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 6.50 GB/s
Size 2097152 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 12.94 GB/s
Size 4194304 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 25.41 GB/s
Size 8388608 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 47.54 GB/s
Size 16777216 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 61.66 GB/s
Size 33554432 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 63.65 GB/s
Size 67108864 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 64.42 GB/s
Size 134217728 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 65.71 GB/s
Size 268435456 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 66.44 GB/s
Size 536870912 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 67.06 GB/s
Size 1073741824 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 67.43 GB/s
Size 2147483648 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 67.67 GB/s
Size 16 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.13 GB/s
Size 16384 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.25 GB/s
Size 32768 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.50 GB/s
Size 65536 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.98 GB/s
Size 131072 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 1.97 GB/s
Size 262144 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 3.83 GB/s
Size 524288 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 7.53 GB/s
Size 1048576 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 14.79 GB/s
Size 2097152 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 30.08 GB/s
Size 4194304 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 62.23 GB/s
Size 8388608 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 121.04 GB/s
Size 16777216 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 213.96 GB/s
Size 33554432 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 230.76 GB/s
Size 67108864 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 229.13 GB/s
Size 134217728 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 231.04 GB/s
Size 268435456 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 232.43 GB/s
Size 536870912 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 233.01 GB/s
Size 1073741824 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 233.25 GB/s
Size 2147483648 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 233.34 GB/s
Size 16 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.23 GB/s
Size 32768 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.49 GB/s
Size 65536 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.98 GB/s
Size 131072 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 1.92 GB/s
Size 262144 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 4.02 GB/s
Size 524288 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 7.58 GB/s
Size 1048576 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 15.45 GB/s
Size 2097152 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 31.14 GB/s
Size 4194304 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 64.28 GB/s
Size 8388608 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 124.26 GB/s
Size 16777216 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 240.90 GB/s
Size 33554432 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 284.94 GB/s
Size 67108864 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 289.09 GB/s
Size 134217728 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 292.09 GB/s
Size 268435456 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 294.29 GB/s
Size 536870912 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 295.46 GB/s
Size 1073741824 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 296.07 GB/s
Size 2147483648 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 296.41 GB/s
Size 16 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.25 GB/s
Size 32768 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.44 GB/s
Size 65536 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.98 GB/s
Size 131072 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 1.93 GB/s
Size 262144 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 3.87 GB/s
Size 524288 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 7.84 GB/s
Size 1048576 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 15.80 GB/s
Size 2097152 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 31.38 GB/s
Size 4194304 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 61.73 GB/s
Size 8388608 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 122.82 GB/s
Size 16777216 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 240.25 GB/s
Size 33554432 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 307.75 GB/s
Size 67108864 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 309.57 GB/s
Size 134217728 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 315.01 GB/s
Size 268435456 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 319.67 GB/s
Size 536870912 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 321.68 GB/s
Size 1073741824 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 322.80 GB/s
Size 2147483648 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 322.76 GB/s
Size 16 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.25 GB/s
Size 32768 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.50 GB/s
Size 65536 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.93 GB/s
Size 131072 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 1.91 GB/s
Size 262144 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 3.99 GB/s
Size 524288 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 7.82 GB/s
Size 1048576 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 15.00 GB/s
Size 2097152 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 30.53 GB/s
Size 4194304 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 60.23 GB/s
Size 8388608 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 121.41 GB/s
Size 16777216 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 249.86 GB/s
Size 33554432 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 316.51 GB/s
Size 67108864 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 319.34 GB/s
Size 134217728 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 324.83 GB/s
Size 268435456 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 328.16 GB/s
Size 536870912 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 329.60 GB/s
Size 1073741824 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 330.43 GB/s
Size 2147483648 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 330.93 GB/s
Size 16 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.24 GB/s
Size 32768 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.50 GB/s
Size 65536 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.97 GB/s
Size 131072 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 2.01 GB/s
Size 262144 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 3.93 GB/s
Size 524288 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 7.54 GB/s
Size 1048576 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 15.38 GB/s
Size 2097152 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 29.77 GB/s
Size 4194304 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 59.52 GB/s
Size 8388608 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 120.42 GB/s
Size 16777216 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 240.83 GB/s
Size 33554432 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 319.10 GB/s
Size 67108864 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 321.93 GB/s
Size 134217728 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 329.75 GB/s
Size 268435456 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 334.00 GB/s
Size 536870912 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 336.12 GB/s
Size 1073741824 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 337.06 GB/s
Size 2147483648 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 337.49 GB/s
"""


raw_data_send = """
Size 16 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.18 GB/s
Size 16384 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.37 GB/s
Size 32768 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 0.74 GB/s
Size 65536 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 1.47 GB/s
Size 131072 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 2.89 GB/s
Size 262144 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 6.11 GB/s
Size 524288 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 11.84 GB/s
Size 1048576 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 22.82 GB/s
Size 2097152 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 47.10 GB/s
Size 4194304 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 90.26 GB/s
Size 8388608 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 178.45 GB/s
Size 16777216 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 193.41 GB/s
Size 33554432 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 197.54 GB/s
Size 67108864 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 199.62 GB/s
Size 134217728 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 202.03 GB/s
Size 268435456 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 203.44 GB/s
Size 536870912 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 204.26 GB/s
Size 1073741824 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 204.38 GB/s
Size 2147483648 bytes, tma=False, num_sms=8, One side transfer Bandwidth: 204.42 GB/s
Size 16 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.10 GB/s
Size 8192 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.18 GB/s
Size 16384 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.38 GB/s
Size 32768 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 0.75 GB/s
Size 65536 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 1.50 GB/s
Size 131072 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 2.96 GB/s
Size 262144 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 6.09 GB/s
Size 524288 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 12.07 GB/s
Size 1048576 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 23.54 GB/s
Size 2097152 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 46.82 GB/s
Size 4194304 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 91.03 GB/s
Size 8388608 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 182.77 GB/s
Size 16777216 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 263.06 GB/s
Size 33554432 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 268.37 GB/s
Size 67108864 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 271.14 GB/s
Size 134217728 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 275.38 GB/s
Size 268435456 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 276.78 GB/s
Size 536870912 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 277.57 GB/s
Size 1073741824 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 278.13 GB/s
Size 2147483648 bytes, tma=False, num_sms=16, One side transfer Bandwidth: 278.26 GB/s
Size 16 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.18 GB/s
Size 16384 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.37 GB/s
Size 32768 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 0.72 GB/s
Size 65536 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 1.48 GB/s
Size 131072 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 2.88 GB/s
Size 262144 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 5.87 GB/s
Size 524288 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 11.79 GB/s
Size 1048576 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 23.15 GB/s
Size 2097152 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 47.08 GB/s
Size 4194304 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 91.37 GB/s
Size 8388608 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 180.04 GB/s
Size 16777216 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 271.96 GB/s
Size 33554432 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 278.88 GB/s
Size 67108864 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 284.19 GB/s
Size 134217728 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 288.56 GB/s
Size 268435456 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 293.12 GB/s
Size 536870912 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 295.11 GB/s
Size 1073741824 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 295.76 GB/s
Size 2147483648 bytes, tma=False, num_sms=32, One side transfer Bandwidth: 296.32 GB/s
Size 16 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.18 GB/s
Size 16384 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.38 GB/s
Size 32768 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 0.73 GB/s
Size 65536 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 1.49 GB/s
Size 131072 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 2.88 GB/s
Size 262144 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 5.94 GB/s
Size 524288 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 11.97 GB/s
Size 1048576 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 23.34 GB/s
Size 2097152 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 47.13 GB/s
Size 4194304 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 94.12 GB/s
Size 8388608 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 183.95 GB/s
Size 16777216 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 273.31 GB/s
Size 33554432 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 309.57 GB/s
Size 67108864 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 313.79 GB/s
Size 134217728 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 319.27 GB/s
Size 268435456 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 322.02 GB/s
Size 536870912 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 323.32 GB/s
Size 1073741824 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 323.82 GB/s
Size 2147483648 bytes, tma=False, num_sms=64, One side transfer Bandwidth: 324.27 GB/s
Size 16 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 512 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.05 GB/s
Size 4096 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.09 GB/s
Size 8192 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.19 GB/s
Size 16384 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.37 GB/s
Size 32768 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 0.69 GB/s
Size 65536 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 1.47 GB/s
Size 131072 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 2.88 GB/s
Size 262144 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 6.00 GB/s
Size 524288 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 11.94 GB/s
Size 1048576 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 24.30 GB/s
Size 2097152 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 45.01 GB/s
Size 4194304 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 89.89 GB/s
Size 8388608 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 183.75 GB/s
Size 16777216 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 284.82 GB/s
Size 33554432 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 315.37 GB/s
Size 67108864 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 320.88 GB/s
Size 134217728 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 328.25 GB/s
Size 268435456 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 332.65 GB/s
Size 536870912 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 334.80 GB/s
Size 1073741824 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 336.00 GB/s
Size 2147483648 bytes, tma=False, num_sms=128, One side transfer Bandwidth: 336.48 GB/s
Size 16 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.01 GB/s
Size 2048 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.24 GB/s
Size 32768 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.50 GB/s
Size 65536 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 0.94 GB/s
Size 131072 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 2.00 GB/s
Size 262144 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 3.77 GB/s
Size 524288 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 7.66 GB/s
Size 1048576 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 15.67 GB/s
Size 2097152 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 31.30 GB/s
Size 4194304 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 61.86 GB/s
Size 8388608 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 112.93 GB/s
Size 16777216 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 219.52 GB/s
Size 33554432 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 231.20 GB/s
Size 67108864 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 229.32 GB/s
Size 134217728 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 232.02 GB/s
Size 268435456 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 233.32 GB/s
Size 536870912 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 234.13 GB/s
Size 1073741824 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 234.38 GB/s
Size 2147483648 bytes, tma=True, num_sms=8, One side transfer Bandwidth: 234.59 GB/s
Size 16 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.24 GB/s
Size 32768 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.47 GB/s
Size 65536 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 0.94 GB/s
Size 131072 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 1.87 GB/s
Size 262144 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 3.87 GB/s
Size 524288 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 7.59 GB/s
Size 1048576 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 15.38 GB/s
Size 2097152 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 29.09 GB/s
Size 4194304 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 61.35 GB/s
Size 8388608 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 113.96 GB/s
Size 16777216 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 243.87 GB/s
Size 33554432 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 309.98 GB/s
Size 67108864 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 309.47 GB/s
Size 134217728 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 315.11 GB/s
Size 268435456 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 318.24 GB/s
Size 536870912 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 319.86 GB/s
Size 1073741824 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 319.99 GB/s
Size 2147483648 bytes, tma=True, num_sms=16, One side transfer Bandwidth: 320.74 GB/s
Size 16 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.24 GB/s
Size 32768 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.46 GB/s
Size 65536 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 0.97 GB/s
Size 131072 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 1.98 GB/s
Size 262144 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 3.89 GB/s
Size 524288 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 7.42 GB/s
Size 1048576 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 14.12 GB/s
Size 2097152 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 30.65 GB/s
Size 4194304 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 61.88 GB/s
Size 8388608 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 122.08 GB/s
Size 16777216 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 241.60 GB/s
Size 33554432 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 329.84 GB/s
Size 67108864 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 333.90 GB/s
Size 134217728 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 340.07 GB/s
Size 268435456 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 344.26 GB/s
Size 536870912 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 347.15 GB/s
Size 1073741824 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 347.76 GB/s
Size 2147483648 bytes, tma=True, num_sms=32, One side transfer Bandwidth: 347.94 GB/s
Size 16 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.01 GB/s
Size 2048 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.25 GB/s
Size 32768 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.48 GB/s
Size 65536 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 0.98 GB/s
Size 131072 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 1.88 GB/s
Size 262144 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 3.78 GB/s
Size 524288 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 7.79 GB/s
Size 1048576 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 15.40 GB/s
Size 2097152 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 30.91 GB/s
Size 4194304 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 60.44 GB/s
Size 8388608 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 120.53 GB/s
Size 16777216 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 239.19 GB/s
Size 33554432 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 348.32 GB/s
Size 67108864 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 353.15 GB/s
Size 134217728 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 358.07 GB/s
Size 268435456 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 362.36 GB/s
Size 536870912 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 364.68 GB/s
Size 1073741824 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 365.50 GB/s
Size 2147483648 bytes, tma=True, num_sms=64, One side transfer Bandwidth: 365.78 GB/s
Size 16 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 64 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 128 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 256 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.00 GB/s
Size 512 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.01 GB/s
Size 1024 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.02 GB/s
Size 2048 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.03 GB/s
Size 4096 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.06 GB/s
Size 8192 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.12 GB/s
Size 16384 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.24 GB/s
Size 32768 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.50 GB/s
Size 65536 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 0.99 GB/s
Size 131072 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 1.93 GB/s
Size 262144 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 3.85 GB/s
Size 524288 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 7.46 GB/s
Size 1048576 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 15.47 GB/s
Size 2097152 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 30.49 GB/s
Size 4194304 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 60.69 GB/s
Size 8388608 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 122.97 GB/s
Size 16777216 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 226.45 GB/s
Size 33554432 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 352.20 GB/s
Size 67108864 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 355.62 GB/s
Size 134217728 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 365.28 GB/s
Size 268435456 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 370.22 GB/s
Size 536870912 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 372.72 GB/s
Size 1073741824 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 374.20 GB/s
Size 2147483648 bytes, tma=True, num_sms=128, One side transfer Bandwidth: 374.98 GB/s"""
# Parse and store data
# Structure: (tma_status, num_sms) -> {'sizes': [], 'bw': []}
data_store = {}

for line in raw_data_reduce.strip().split('\n'):
    match = re.search(r'Size (\d+) bytes, tma=(True|False), num_sms=(\d+), One side transfer Bandwidth: ([\d.]+) GB/s', line)
    if match:
        size = int(match.group(1))
        tma = match.group(2) == 'True'
        num_sms = int(match.group(3))
        bw = float(match.group(4))
        
        key = (tma, num_sms)
        if key not in data_store:
            data_store[key] = {'sizes': [], 'bw': []}
        
        data_store[key]['sizes'].append(size)
        data_store[key]['bw'].append(bw)

# Configure color map
sms_counts = sorted(list(set(k[1] for k in data_store.keys())))
colors = plt.cm.viridis([i/len(sms_counts) for i in range(len(sms_counts))])
sms_color_map = {sms: color for sms, color in zip(sms_counts, colors)}

plt.figure(figsize=(12, 7))

sorted_keys = sorted(data_store.keys(), key=lambda k: (k[1], k[0]))

for tma, num_sms in sorted_keys:
    values = data_store[(tma, num_sms)]
    label = f'SMS={num_sms}, TMA={tma}'
    linestyle = '--' if tma else '-' 
    color = sms_color_map[num_sms]
    
    sorted_pairs = sorted(zip(values['sizes'], values['bw']))
    sizes = [p[0] for p in sorted_pairs]
    bw = [p[1] for p in sorted_pairs]
    
    plt.plot(sizes, bw, label=label, linestyle=linestyle, color=color, linewidth=2, alpha=0.9)

# --- Key change: use a log2 X axis ---
plt.xscale('log', base=2)

# Format X axis ticks with readable units
def human_readable_size(x, pos):
    if x >= 1024**3:
        return f'{x/1024**3:.0f}GB'
    elif x >= 1024**2:
        return f'{x/1024**2:.0f}MB'
    elif x >= 1024:
        return f'{x/1024:.0f}KB'
    else:
        return f'{x:.0f}B'

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(human_readable_size))

plt.title('Bandwidth vs Transfer Size (Log Scale)', fontsize=15)
plt.xlabel('Transfer Size (Log Scale)', fontsize=12)
plt.ylabel('Bandwidth (GB/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, which="both") # which="both" ensures minor grid lines are visible on the log axis

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Configuration")

plt.tight_layout()
plt.savefig('sm_throughput_log_scale.png', dpi=300)

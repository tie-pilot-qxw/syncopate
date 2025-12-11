import matplotlib.pyplot as plt
import re
import matplotlib.ticker as ticker

# Raw measurement data
raw_data = """
Size 8 bytes, One side transfer Bandwidth: 0.00 GB/s
Size 16 bytes, One side transfer Bandwidth: 0.00 GB/s
Size 32 bytes, One side transfer Bandwidth: 0.01 GB/s
Size 64 bytes, One side transfer Bandwidth: 0.02 GB/s
Size 128 bytes, One side transfer Bandwidth: 0.04 GB/s
Size 256 bytes, One side transfer Bandwidth: 0.08 GB/s
Size 512 bytes, One side transfer Bandwidth: 0.20 GB/s
Size 1024 bytes, One side transfer Bandwidth: 0.42 GB/s
Size 2048 bytes, One side transfer Bandwidth: 0.83 GB/s
Size 4096 bytes, One side transfer Bandwidth: 1.79 GB/s
Size 8192 bytes, One side transfer Bandwidth: 3.62 GB/s
Size 16384 bytes, One side transfer Bandwidth: 4.90 GB/s
Size 32768 bytes, One side transfer Bandwidth: 9.71 GB/s
Size 65536 bytes, One side transfer Bandwidth: 19.58 GB/s
Size 131072 bytes, One side transfer Bandwidth: 39.26 GB/s
Size 262144 bytes, One side transfer Bandwidth: 76.68 GB/s
Size 524288 bytes, One side transfer Bandwidth: 153.11 GB/s
Size 1048576 bytes, One side transfer Bandwidth: 229.00 GB/s
Size 2097152 bytes, One side transfer Bandwidth: 284.53 GB/s
Size 4194304 bytes, One side transfer Bandwidth: 305.59 GB/s
Size 8388608 bytes, One side transfer Bandwidth: 352.11 GB/s
Size 16777216 bytes, One side transfer Bandwidth: 370.02 GB/s
Size 33554432 bytes, One side transfer Bandwidth: 381.54 GB/s
Size 67108864 bytes, One side transfer Bandwidth: 388.77 GB/s
Size 134217728 bytes, One side transfer Bandwidth: 394.06 GB/s
Size 268435456 bytes, One side transfer Bandwidth: 396.33 GB/s
Size 536870912 bytes, One side transfer Bandwidth: 397.76 GB/s
Size 1073741824 bytes, One side transfer Bandwidth: 398.16 GB/s
"""

sizes = []
bandwidths = []

# Parse data
for line in raw_data.strip().split('\n'):
    match = re.search(r'Size (\d+) bytes.*Bandwidth: ([\d.]+) GB/s', line)
    if match:
        sizes.append(int(match.group(1)))
        bandwidths.append(float(match.group(2)))

# --- Plot ---
plt.figure(figsize=(12, 6))

# Plot with a linear scale
# Use a small marker to keep the dense points on the left readable
plt.plot(sizes, bandwidths, marker='.', linewidth=2, color='#1f77b4', markersize=8)

# Explicitly use linear scale for clarity
plt.xscale('linear')

# Format X axis labels as GB
def gb_formatter(x, pos):
    gb_val = x / (1024**3)
    return f'{gb_val:.1f} GB'

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(gb_formatter))

# Set tick interval to 1GB
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1024**3))

plt.title('Memory Transfer Bandwidth (Linear Scale)', fontsize=14)
plt.xlabel('Transfer Size (Linear)', fontsize=12)
plt.ylabel('Bandwidth (GB/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Example annotation for the saturation region (around 2MB / index 18)
# saturation_idx = 18
# plt.annotate('Bandwidth Saturation Region',
#              xy=(sizes[saturation_idx], bandwidths[saturation_idx]),
#              xytext=(sizes[saturation_idx] + 1024**3, bandwidths[saturation_idx] - 50),
#              arrowprops=dict(facecolor='black', arrowstyle='->'),
#              fontsize=10)

plt.tight_layout()
plt.savefig('throughput_linear.png', dpi=300)

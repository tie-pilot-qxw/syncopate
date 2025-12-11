import matplotlib.pyplot as plt
import math

def d2xy(d, k):
    """
    Convert distance d on a 1D Hilbert curve to (x, y) on a 2^k x 2^k grid.

    Args:
        d (int): distance along the curve (0 <= d < 4**k).
        k (int): curve order (grid side length is 2**k).

    Returns:
        tuple: (x, y) coordinates.
    """
    n = 2**k
    x, y = 0, 0
    s = 1
    while s < n:
        rx = 1 & (d >> 1)
        ry = 1 & (d ^ rx)
        
        # Rotation and flip logic
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x # swap x and y
            
        x += s * rx
        y += s * ry
        d >>= 2
        s *= 2
    return x, y

def draw_hilbert_mn(m, n):
    """
    Generate and plot a Hilbert curve for an arbitrary m x n grid.
    
    Args:
        m (int): grid width (columns).
        n (int): grid height (rows).
    """
    if m <= 0 or n <= 0:
        print("Error: m and n must be positive integers.")
        return

    # 1. Find the smallest 2^k x 2^k grid that can contain m x n
    side = max(m, n)
    # math.log2(side) computes bits to represent side-1
    # math.ceil rounds up to get the order k
    k = math.ceil(math.log2(side)) if side > 1 else 1
    N = 2**k # side length of the larger grid
    
    print(f"Target grid: {m}x{n}")
    print(f"Use a {N}x{N} (order k={k}) Hilbert curve to cover the grid.")

    # 2. Generate all points on the full 2^k x 2^k curve
    points = []
    total_points_in_large_square = 4**k
    for d in range(total_points_in_large_square):
        x, y = d2xy(d, k)
        # 3. Keep only the points that fall in the m x n region
        if x < m and y < n:
            points.append((x, y))

    if not points:
        print("No points found inside the specified m x n region.")
        return

    # 4. Prepare for plotting
    # Shift coordinates to cell centers for clarity
    x_coords = [p[0] + 0.5 for p in points]
    y_coords = [p[1] + 0.5 for p in points]

    fig, ax = plt.subplots(figsize=(m / 2 + 1, n / 2 + 1))
    
    # Plot curve
    ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='royalblue', markersize=4)

    # Beautify plot
    ax.set_title(f'Hilbert Curve for a {m}x{n} Grid (on a {N}x{N} base)')
    ax.set_xlabel('M (Width)')
    ax.set_ylabel('N (Height)')
    
    # Grid lines and ticks
    ax.set_xticks([i for i in range(m + 1)])
    ax.set_yticks([i for i in range(n + 1)])
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Force square cells
    ax.set_aspect('equal', adjustable='box')
    
    # Flip y-axis so (0,0) appears at top-left like matrix indexing
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- You can tweak M and N here for quick experiments ---

    # Example 1: classic 4x4 grid (2^2 x 2^2)
    print("--- Example 1: 4x4 grid ---")
    draw_hilbert_mn(m=4, n=4)

    # Example 2: non-square 7x5 grid
    print("\n--- Example 2: 7x5 grid ---")
    draw_hilbert_mn(m=7, n=5)
    
    # Example 3: 6x6 grid (not a power of two)
    print("\n--- Example 3: 6x6 grid ---")
    draw_hilbert_mn(m=6, n=6)

    # Example 4: rectangular 15x3 grid
    print("\n--- Example 4: 15x3 grid ---")
    draw_hilbert_mn(m=15, n=3)

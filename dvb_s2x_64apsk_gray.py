import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def generate_dvb_s2x_64apsk_constellation():
    """
    Generates the DVB-S2X 64-APSK constellation points (8+16+20+20 configuration).
    Radii ratios: R2/R1 = 2.2, R3/R1 = 3.6, R4/R1 = 5.2
    """
    gamma = np.array([1.0, 2.2, 3.6, 5.2])
    M_rings = np.array([8, 16, 20, 20])
    
    energy_unscaled = np.sum(M_rings * (gamma ** 2))
    R1 = np.sqrt(64 / energy_unscaled)
    
    radii = gamma * R1
    
    points = []
    
    # Generate points ring by ring (spiral order)
    # Ring 1
    for i in range(M_rings[0]):
        theta = 2 * np.pi * i / M_rings[0]
        points.append(radii[0] * np.exp(1j * theta))
        
    # Ring 2
    for i in range(M_rings[1]):
        theta = 2 * np.pi * i / M_rings[1]
        points.append(radii[1] * np.exp(1j * theta))
        
    # Ring 3
    for i in range(M_rings[2]):
        theta = 2 * np.pi * i / M_rings[2]
        points.append(radii[2] * np.exp(1j * theta))
        
    # Ring 4
    for i in range(M_rings[3]):
        theta = 2 * np.pi * i / M_rings[3]
        points.append(radii[3] * np.exp(1j * theta))
        
    return np.array(points)

def gray_code(n):
    """Computes the Gray code of integer n."""
    return n ^ (n >> 1)

def inverse_gray_code(n):
    """Computes the inverse Gray code of integer n."""
    mask = n >> 1
    while mask != 0:
        n = n ^ mask
        mask = mask >> 1
    return n

def count_set_bits(n):
    """Counts set bits in integer n (population count)."""
    return bin(n).count('1')

def perform_simulation(constellation, snr_db_range, use_gray=False, num_symbols=100000):
    M = len(constellation)
    k = int(np.log2(M)) # 6 bits
    
    # 1. Generate Data (integers 0..63)
    # These represent the 6-bit messages we want to send
    data_int = np.random.randint(0, M, num_symbols)
    
    # 2. Map to Constellation Indices
    if use_gray:
        # We want the Label(i) to be Gray(i).
        # So transmission symbol index i for message m should satisfy Gray(i) = m.
        # Thus i = InverseGray(m).
        # Vectorized inverse gray? Loop for simplicity or numpy map
        # Since M is small (64), we can precompute the table.
        inv_gray_table = np.array([inverse_gray_code(i) for i in range(M)])
        tx_indices = inv_gray_table[data_int]
    else:
        # Natural Binary: Label(i) = i.
        # Transmission symbol index i for message m satisfies i = m.
        tx_indices = data_int
        
    tx_symbols = constellation[tx_indices]
    
    ber_results = []
    ser_results = []
    
    mode_name = "Gray" if use_gray else "No-Gray"
    print(f"Simulating {mode_name} for SNR range: {snr_db_range} dB")
    
    # Precompute Gray table for decoding
    gray_table = np.array([gray_code(i) for i in range(M)])
    
    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db / 10.0)
        n0 = 1.0 / snr_linear
        noise_std = np.sqrt(n0 / 2)
        
        noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        rx_symbols = tx_symbols + noise
        
        # Demap (European Distance)
        rx_coords = np.column_stack((rx_symbols.real, rx_symbols.imag))
        const_coords = np.column_stack((constellation.real, constellation.imag))
        
        dists = distance.cdist(rx_coords, const_coords, 'euclidean')
        estimated_indices = np.argmin(dists, axis=1)
        
        # Recover Message Bits (Integers)
        if use_gray:
            # We detected symbol index i.
            # The label (message) is Gray(i).
            estimated_message_int = gray_table[estimated_indices]
        else:
            # Natural: Label(i) = i.
            estimated_message_int = estimated_indices
        
        # Calculate Errors
        # SER: Message mismatch
        symbol_errors = np.sum(data_int != estimated_message_int)
        ser = symbol_errors / num_symbols
        ser_results.append(ser)
        
        # BER: Hamming distance between transmitted and estimated messages
        # bit_diffs = data_int ^ estimated_message_int
        # total_bit_errors = sum(popcount(x))
        bit_diffs = data_int ^ estimated_message_int
        # Vectorized popcount for simple approach: using map or bitwise ops
        # Since max 6 bits, simple loop or bitwise logic
        # np.bitwise_count is available in newer numpy, but for safety:
        # Manual bit counting for 6 bits
        c = 0
        temp_diffs = bit_diffs
        for _ in range(k):
            c += (temp_diffs & 1)
            temp_diffs >>= 1
            
        total_bit_errors = np.sum(c)
        ber = total_bit_errors / (num_symbols * k)
        ber_results.append(ber)
        
        print(f"SNR: {snr_db} dB, BER: {ber:.2e}, SER: {ser:.2e}")
        
    return ber_results, ser_results

def main():
    constellation = generate_dvb_s2x_64apsk_constellation()
    
    # Plot Constellation with labelling (optional, but good for verify)
    # Let's Skip label plotting to keep it simple, concentrate on graphs.
    
    snr_range = np.arange(10, 26, 1)
    
    # Simulation: Natural (No Gray)
    ber_nat, ser_nat = perform_simulation(constellation, snr_range, use_gray=False)
    
    # Simulation: Gray
    ber_gray, ser_gray = perform_simulation(constellation, snr_range, use_gray=True)
    
    # Plot BER
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_nat, 'bo--', label='BER (No Gray)')
    plt.semilogy(snr_range, ber_gray, 'rs-', label='BER (Gray)')
    plt.title('BER: 64-APSK Gray vs No Gray')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.savefig('64apsk_gray_ber.png')
    
    # Plot SER
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ser_nat, 'bo--', label='SER (No Gray)')
    plt.semilogy(snr_range, ser_gray, 'rs-', label='SER (Gray)')
    plt.title('SER: 64-APSK Gray vs No Gray')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.savefig('64apsk_gray_ser.png')
    
    print("Plots saved to 64apsk_gray_ber.png and 64apsk_gray_ser.png")
    
    # Plot Constellation with Labels
    # Natural Mapping
    labels_nat = [f"{i:06b}" for i in range(len(constellation))]
    plot_constellation_with_labels(constellation, labels_nat, "64-APSK Natural Mapping", "64apsk_natural_constellation.png")
    
    # Gray Mapping
    gray_labels_int = [gray_code(i) for i in range(len(constellation))]
    labels_gray = [f"{i:06b}" for i in gray_labels_int] # These are the Gray codewords assigned to point i
    
    plot_constellation_with_labels(constellation, labels_gray, "64-APSK Gray Mapping", "64apsk_gray_constellation.png", reference_labels=labels_nat)
    
    print("Constellation plots saved with labels.")

def plot_constellation_with_labels(points, labels, title, filename, reference_labels=None):
    plt.figure(figsize=(12, 12))
    
    # Base plot (we will redraw points with specific colors in the loop)
    # plt.scatter(points.real, points.imag, c='blue', marker='.', alpha=0.5)
    
    # Annotate points
    for i, point in enumerate(points):
        label = labels[i]
        
        # Determine color
        color = 'black'
        point_color = 'blue'
        
        if reference_labels is not None:
            if label != reference_labels[i]:
                color = 'red'
                point_color = 'red'
        
        # Plot point
        plt.scatter(point.real, point.imag, c=point_color, marker='o', alpha=0.6)
        
        # Plot label
        plt.text(point.real, point.imag, label, fontsize=8, ha='center', va='bottom', alpha=0.9, color=color, fontweight='bold' if color=='red' else 'normal')
        
    plt.title(title)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    main()

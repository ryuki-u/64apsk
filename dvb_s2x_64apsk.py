import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def generate_dvb_s2x_64apsk_constellation():
    """
    Generates the DVB-S2X 64-APSK constellation points (8+16+20+20 configuration).
    Radii ratios: R2/R1 = 2.2, R3/R1 = 3.6, R4/R1 = 5.2
    
    Returns:
        points (np.array): Complex array of constellation points.
    """
    # Radii ratios from DVB-S2X standard (approximate for 7/9 modcod)
    gamma = np.array([1.0, 2.2, 3.6, 5.2])
    
    # Normalize radii so average power is 1 (optional, but good for consistent SNR)
    # Number of points per ring: 8, 16, 20, 20
    M_rings = np.array([8, 16, 20, 20])
    
    # Determine base radius R1 such that total energy is normalized to 1 per symbol
    # Energy = sum(Ni * Ri^2) / sum(Ni)
    energy_unscaled = np.sum(M_rings * (gamma ** 2))
    R1 = np.sqrt(64 / energy_unscaled)
    
    radii = gamma * R1
    
    points = []
    
    # Ring 1: 8 points
    # Phase offset typically 0 for first ring (or optimized)
    phase_off_1 = 0 
    for i in range(M_rings[0]):
        theta = 2 * np.pi * i / M_rings[0] + phase_off_1
        points.append(radii[0] * np.exp(1j * theta))
        
    # Ring 2: 16 points
    phase_off_2 = 0 # Stagger not strictly defined without table, assuming aligned for simplicity or slight offset
    for i in range(M_rings[1]):
        theta = 2 * np.pi * i / M_rings[1] + phase_off_2
        points.append(radii[1] * np.exp(1j * theta))
        
    # Ring 3: 20 points
    phase_off_3 = 0
    for i in range(M_rings[2]):
        theta = 2 * np.pi * i / M_rings[2] + phase_off_3
        points.append(radii[2] * np.exp(1j * theta))
        
    # Ring 4: 20 points
    phase_off_4 = 0
    for i in range(M_rings[3]):
        theta = 2 * np.pi * i / M_rings[3] + phase_off_4
        points.append(radii[3] * np.exp(1j * theta))
        
    return np.array(points)

def plot_constellation(points):
    """
    Plots the constellation in the complex plane.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(points.real, points.imag, c='blue', marker='.', label='Constellation Points')
    
    # Circles removed as requested
        
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('DVB-S2X 64-APSK Constellation (8+16+20+20)')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()

def perform_ber_simulation(constellation, snr_db_range, num_symbols=50000):
    """
    Simulates BER vs SNR for the given constellation.
    Uses a simple nearest-neighbor demapper and assumes a random bit mapping 
    (since exact DVB-S2X mapping is complex).
    
    Note: Real DVB-S2X uses specific bit mappings to optimize BER. 
    This simulation approximates performance.
    """
    M = len(constellation)
    k = int(np.log2(M)) # bits per symbol
    
    # Generate random bits
    num_bits = num_symbols * k
    tx_bits = np.random.randint(0, 2, num_bits)
    
    # Convert bits to symbol indices
    # Reshape bits to (num_symbols, k)
    tx_bits_reshaped = tx_bits.reshape((num_symbols, k))
    
    # Simple binary to decimal conversion for mapping (Not Gray coded)
    # To improve, one could implement a Gray code generator, but for 64-APSK generic is tricky.
    # We will use a random fixed mapping for consistency in this demo, 
    # but strictly speaking this makes BER worse than optimal.
    # To mitigate, let's just map indices 0..63 directly to the points array order.
    tx_indices = np.zeros(num_symbols, dtype=int)
    for i in range(k):
        tx_indices += tx_bits_reshaped[:, i] * (2**i)
        
    # Valid indices check (should be fine if M=64)
    tx_indices = tx_indices % M
    
    tx_symbols = constellation[tx_indices]
    
    ber_results = []
    
    print(f"Simulating BER for SNR range: {snr_db_range} dB")
    
    for snr_db in snr_db_range:
        # Calculate noise power
        # SNR = Es / N0
        # Since Es is normalized to 1, N0 = 1 / (10^(SNR/10))
        # Noise is complex gaussian
        snr_linear = 10**(snr_db / 10.0)
        n0 = 1.0 / snr_linear
        noise_std = np.sqrt(n0 / 2)
        
        noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        rx_symbols = tx_symbols + noise
        
        # Demap (ML detection / Minimum Euclidean Distance)
        # Compute distances from every rx symbol to every constellation point
        # This can be slow for large N, but efficient with broadcasting or cdist
        
        # Efficient way using scipy cdist
        # rx_points: (N, 2), const_points: (M, 2)
        rx_coords = np.column_stack((rx_symbols.real, rx_symbols.imag))
        const_coords = np.column_stack((constellation.real, constellation.imag))
        
        dists = distance.cdist(rx_coords, const_coords, 'euclidean')
        estimated_indices = np.argmin(dists, axis=1)
        
        # Recover bits from indices
        rx_bits = np.zeros(num_bits, dtype=int)
        estimated_bits_reshaped = np.zeros((num_symbols, k), dtype=int)
        
        for i in range(k):
             # Extract ith bit from integer
             estimated_bits_reshaped[:, i] = (estimated_indices >> i) & 1
             
        rx_bits = estimated_bits_reshaped.flatten()
        
        # Count errors
        bit_errors = np.sum(tx_bits != rx_bits)
        ber = bit_errors / num_bits
        ber_results.append(ber)
        
        print(f"SNR: {snr_db} dB, BER: {ber:.2e}")
        
    return ber_results

def generate_64qam_constellation():
    """
    Generates a standard square 64-QAM constellation.
    Normalized to average power = 1.
    """
    # M-QAM levels: -(sqrt(M)-1), ..., -1, 1, ..., sqrt(M)-1
    # For 64-QAM: -7, -5, -3, -1, 1, 3, 5, 7
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    
    points = []
    for I in levels:
        for Q in levels:
            points.append(I + 1j*Q)
            
    points = np.array(points)
    
    # Normalize energy
    # Average energy for square M-QAM with ints is 2*(M-1)/3
    # For M=64: 2*63/3 = 42
    avg_energy = 42.0
    scale_factor = np.sqrt(avg_energy)
    
    return points / scale_factor

def generate_dvb_s2x_64apsk_8_12_20_24_constellation():
    """
    Generates the DVB-S2X 64-APSK constellation points (8+12+20+24 configuration).
    Radii ratios are approximated or based on typical APSK spacing.
    Assumed Radii ratios: R2/R1 = 2.4, R3/R1 = 3.8, R4/R1 = 5.2
    """
    # Radii ratios (Approximated for reasonable distribution)
    gamma = np.array([1.0, 2.4, 3.8, 5.2])
    
    # Number of points per ring: 8, 12, 20, 24
    M_rings = np.array([8, 12, 20, 24])
    
    # Determine base radius R1 such that total energy is normalized to 1 per symbol
    energy_unscaled = np.sum(M_rings * (gamma ** 2))
    R1 = np.sqrt(64 / energy_unscaled)
    
    radii = gamma * R1
    
    points = []
    
    # Ring 1: 8 points
    phase_off_1 = 0 
    for i in range(M_rings[0]):
        theta = 2 * np.pi * i / M_rings[0] + phase_off_1
        points.append(radii[0] * np.exp(1j * theta))
        
    # Ring 2: 12 points
    phase_off_2 = 0 
    for i in range(M_rings[1]):
        theta = 2 * np.pi * i / M_rings[1] + phase_off_2
        points.append(radii[1] * np.exp(1j * theta))
        
    # Ring 3: 20 points
    phase_off_3 = 0
    for i in range(M_rings[2]):
        theta = 2 * np.pi * i / M_rings[2] + phase_off_3
        points.append(radii[2] * np.exp(1j * theta))
        
    # Ring 4: 24 points
    phase_off_4 = 0
    for i in range(M_rings[3]):
        theta = 2 * np.pi * i / M_rings[3] + phase_off_4
        points.append(radii[3] * np.exp(1j * theta))
        
    return np.array(points)

def main():
    # 1. Generate Constellations
    constellation = generate_dvb_s2x_64apsk_constellation()
    constellation_8_12_20_24 = generate_dvb_s2x_64apsk_8_12_20_24_constellation()
    
    # 2. Plot Constellations
    # Original
    plot_constellation(constellation)
    plt.title('DVB-S2X 64-APSK Constellation (8+16+20+20)')
    plt.savefig('dvb_s2x_64apsk_constellation.png') 
    print("Constellation plot saved to dvb_s2x_64apsk_constellation.png")
    
    # New 8-12-20-24
    plot_constellation(constellation_8_12_20_24)
    plt.title('DVB-S2X 64-APSK Constellation (8+12+20+24)')
    plt.savefig('dvb_s2x_64apsk_8_12_20_24_constellation.png')
    print("Constellation plot saved to dvb_s2x_64apsk_8_12_20_24_constellation.png")
    
    # 3. Simulate BER
    snr_range = np.arange(10, 26, 1) # Scan 10dB to 25dB
    
    # Run 64APSK (8+16+20+20) Simulation
    print("Starting 64-APSK (8+16+20+20) Simulation...")
    ber_apsk = perform_ber_simulation(constellation, snr_range)
    
    # Run 64APSK (8+12+20+24) Simulation
    print("Starting 64-APSK (8+12+20+24) Simulation...")
    ber_apsk_new = perform_ber_simulation(constellation_8_12_20_24, snr_range)
    
    # Run 64QAM Simulation
    print("Starting 64-QAM Simulation...")
    constellation_qam = generate_64qam_constellation()
    ber_qam = perform_ber_simulation(constellation_qam, snr_range)
    
    # 4. Plot BER
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_apsk, 'bo-', linewidth=2, label='64-APSK (8-16-20-20)')
    plt.semilogy(snr_range, ber_apsk_new, 'gx--', linewidth=2, label='64-APSK (8-12-20-24)')
    plt.semilogy(snr_range, ber_qam, 'rs-.', linewidth=2, label='64-QAM (Standard)')
    plt.title('BER vs SNR Comparison')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which="both", ls="--")
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.savefig('dvb_s2x_64apsk_ber.png')
    print("BER plot saved to dvb_s2x_64apsk_ber.png")
    # plt.show()

if __name__ == "__main__":
    main()

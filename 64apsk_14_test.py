# ==========================================
# 1. Constellation Generation Functions
# ==========================================

def generate_apsk_custom(ring_counts, alpha):
    """
    Generates a 64-APSK constellation with specified ring counts and radius spacing factor alpha.
    
    Parameters:
        ring_counts (list): List of number of points per ring (e.g., [4, 12, 20, 28]).
        alpha (float): Radius spacing factor. Radii will be R1 * [1, 1+alpha, 1+2*alpha, ...].
        
    Returns:
        points (np.array): Normalized complex array of constellation points (Average Power = 1).
    """
    """
    【責務】
    - 64-APSKのコンスタレーション点群を生成する関数。
    - リングごとの点数（ring_counts）と、半径比を決める1パラメータ alpha を入力として、
      各リングの半径を決めた上で、等角配置で点を生成する。
    - 最後に平均電力（平均シンボルエネルギー）が1になるように正規化する。

    【手順書との対応】
    - 「半径比は alpha による1パラメータ化でスイープ」に対応。
      具体的には gamma = [1, 1+α, 1+2α, 1+3α] の形で半径比を作っている。
    - 「平均電力1に正規化（公平比較）」に対応。
    - 「リング内は等角配置」「位相オフセットは0固定」に対応。
    """
    ring_counts = np.array(ring_counts)
    num_rings = len(ring_counts)
    
    # Define relative radii based on alpha
    # gamma = [1, 1+alpha, 1+2alpha, ...]
    gamma = np.array([1.0 + i * alpha for i in range(num_rings)])
    
    # Determine base radius R1 such that total energy is normalized to 1 per symbol
    # Energy = sum(Ni * Ri^2) / sum(Ni)
    # sum(Ni) should be 64 for 64-APSK
    total_points = np.sum(ring_counts)
    energy_unscaled = np.sum(ring_counts * (gamma ** 2))
    R1 = np.sqrt(total_points / energy_unscaled)
    
    radii = gamma * R1
    
    points = []
    
    # Generate points for each ring
    # Using equi-angular spacing with 0 phase offset for all rings (Simplified design)
    for r_idx in range(num_rings):
        n_points = ring_counts[r_idx]
        radius = radii[r_idx]
        phase_offset = 0.0
        
        for i in range(n_points):
            theta = 2 * np.pi * i / n_points + phase_offset
            points.append(radius * np.exp(1j * theta))
            
    return np.array(points)


def generate_64qam_constellation():
    """
    Generates a standard square 64-QAM constellation.
    Normalized to average power = 1.
    """
    """
    【責務】
    - 64-QAM（正方格子）の標準的なコンスタレーション点群を生成する関数。
    - 8x8 = 64点の格子（I/Qレベル）を作り、平均電力1に正規化する。

    【手順書との対応】
    - 「64QAMとの比較（1つだけ）」のための比較対象コンスタレーション生成に対応。
    - 「平均電力1に正規化（公平比較）」に対応。
    """
    # M-QAM levels: -7, -5, -3, -1, 1, 3, 5, 7
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    
    points = []
    for I in levels:
        for Q in levels:
            points.append(I + 1j*Q)
            
    points = np.array(points)
    
    # Normalize energy
    # Avg energy for square 64-QAM with ints is 42
    avg_energy = 42.0
    scale_factor = np.sqrt(avg_energy)
    
    return points / scale_factor


# ==========================================
# 2. Simulation Functions (SER)
# ==========================================

def simulate_ser(constellation, snr_db, num_symbols=100000):
    """
    Simulates Symbol Error Rate (SER) for a given constellation at a specific SNR.
    Uses Nearest Neighbor (ML) decoding.
    """
    """
    【責務】
    - 指定したコンスタレーションについて、AWGN環境でのSERをモンテカルロ法で推定する。
    - 送信 → 雑音付加 → 復調（最近傍判定） → SER計算 を行う。

    【手順書との対応】
    - チャネル：AWGN に対応。
    - 復調：最近傍判定（最小ユークリッド距離、ML復調）に対応。
    - SNR固定1点（例 18 dB）は run_experiment() 側で固定して呼び出される前提。
    - 評価指標：SER（BERではない）に対応。
    """
    M = len(constellation)
    
    # 1. Generate random Indices
    tx_indices = np.random.randint(0, M, num_symbols)
    tx_symbols = constellation[tx_indices]
    
    # 2. Add AWGN
    # SNR = Es / N0, Es = 1 (Normalized)
    # N0 = 1 / 10^(SNR/10)
    # Noise Power = N0.  Complex Noise Std Dev = sqrt(N0/2) per dimension.
    snr_linear = 10**(snr_db / 10.0)
    n0 = 1.0 / snr_linear
    noise_std = np.sqrt(n0 / 2)
    
    noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = tx_symbols + noise
    
    # 3. Demodulate (Nearest Neighbor)
    # Use scipy.spatial.distance.cdist for efficient Euclidean distance calculation
    # rx_points: (N, 2), const_points: (M, 2)
    rx_coords = np.column_stack((rx_symbols.real, rx_symbols.imag))
    const_coords = np.column_stack((constellation.real, constellation.imag))
    
    # dists will be (N, M) matrix
    # For large N, we might want to batch this if memory is an issue, 
    # but for N=100,000 it implies ~6400 * 8 bytes ~ 50MB, which is fine.
    dists = distance.cdist(rx_coords, const_coords, 'euclidean')
    estimated_indices = np.argmin(dists, axis=1)
    
    # 4. Calculate SER
    # Count how many indices do NOT match
    errors = np.sum(tx_indices != estimated_indices)
    ser = errors / num_symbols
    
    return ser


# ==========================================
# 3. Visualization Functions
# ==========================================

def plot_constellation(points, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(points.real, points.imag, c='blue', marker='o', label='Constellation Points')
    
    """
    【責務】
    - コンスタレーション点群（複素平面上の点）を散布図として保存する。
    - 視覚的にリング構造が分かるように、半径ごとの円（リング）も補助的に描画する。

    【手順書との対応】
    - 成果物①「現行64APSKのコンスタレーション図（正規化込み）」に対応。
    - 議論用の図として「どんな配置になっているか」を示す役割。
    """
    
    # Draw rings for visual aid (optional)
    radii = np.unique(np.abs(points))
    for r in radii:
        circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle=':', alpha=0.5)
        plt.gca().add_artist(circle)

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_sweep_results(alpha_values, ser_results_A, ser_results_B, best_A, best_B, filename):
    plt.figure(figsize=(10, 6))

    """
    【責務】
    - α（半径比パラメータ）を横軸、SERを縦軸（対数）にして折れ線グラフを生成する。
    - Config A / Config B を同じ図に重ねて比較できるようにする。
    - さらに最良点（best_A, best_B）をマーカーで強調表示する。

    【手順書との対応】
    - 成果物②「半径比スイープのSERグラフ（SNR固定）」に対応。
    - 「リング構成が違うと最適αが変わる」を示す主要な図になる。
    """
    
    # Plot Config A
    plt.semilogy(alpha_values, ser_results_A, 'b-o', label='Config A (4-12-20-28)')
    # Plot Config B
    plt.semilogy(alpha_values, ser_results_B, 'g-s', label='Config B (8-16-20-20)')
    
    # Mark best points
    plt.plot(best_A[0], best_A[1], 'r*', markersize=15, label=f'Best A (alpha={best_A[0]:.2f})')
    plt.plot(best_B[0], best_B[1], 'm*', markersize=15, label=f'Best B (alpha={best_B[0]:.2f})')
    
    plt.title('SER vs Radius Factor (alpha) at SNR=18dB')
    plt.xlabel('Radius Factor alpha')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_comparison_bar(ser_apsk, ser_qam, label_apsk, filename):
    labels = [label_apsk, '64-QAM']
    values = [ser_apsk, ser_qam]
    
    plt.figure(figsize=(7, 6))

    """
    【責務】
    - 最良64-APSK（または指定したAPSK）と64-QAMのSERを棒グラフで比較する。
    - “同じSNRならどちらがSERが小さいか” を直感的に見せる目的。

    【手順書との対応】
    - 成果物③「64QAM vs 64APSK のSER比較」に対応。
    - 注意：AWGNだけの比較なので、衛星通信の“非線形耐性”の議論には直結しにくいが、
      「AWGN環境ではQAMが有利になりがち」の確認として使える。
    """

    bars = plt.bar(labels, values, color=['blue', 'red'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2e}',
                 ha='center', va='bottom')
        
    plt.title('SER Comparison (SNR=18dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


# ==========================================
# 4. Main Experiment Logic
# ==========================================

def run_experiment():
    print("=== Starting 64-APSK Optimization Experiment ===")

    """
    【責務（この研究の実験プロトコル本体）】
    - 実験条件（SNR固定、AWGN、最近傍判定、平均電力正規化）を固定した上で
      64APSKの半径比パラメータ α をスイープし、SER最小となるαを探索する。
    - さらにリング構成の違い（AとB）がSERに与える影響も同時に比較する。
    - 最後に64QAMと比較し、成果物の図を保存する。

    【手順書との対応（そのまま対応）】
    Step 1: 粗いスイープ（例 0.2〜0.8 を0.1刻み、2e4シンボル）
    Step 2: 良さそうな範囲だけ細かくスイープ（例 0.02刻み、1e5シンボル）
    Step 3: 64QAMとの比較（同一SNRでSER比較）
    Step 4: 成果物作成（図3枚）

    【注意点（手順書と完全一致ではない部分）】
    - Fine Sweep の範囲が A/Bそれぞれの粗探索ベストを中心にするのではなく、
      2つのベストの平均(center_alpha)を中心に同一範囲を探索している。
      もしAとBの最適αが離れていると、片方の最適を取り逃す可能性がある。
    - コンスタレーション図はAのみ描画している（Bも描くとより説明しやすい）。
    - QAM比較もAのみ（Bも比較できるとより完全）。
    """

    # Parameters
    SNR_DB = 18.0
    NUM_SYMBOLS_COARSE = 20000
    NUM_SYMBOLS_FINE   = 100000
    
    # Configs
    CONFIG_A = [4, 12, 20, 28] # Baseline
    CONFIG_B = [8, 16, 20, 20] # Alternative
    
    # --- Step 2: Coarse Sweep ---
    print("\n--- Step 2: Coarse Sweep (alpha 0.2 - 0.8) ---")
    alpha_coarse = np.arange(0.2, 0.81, 0.1)
    results_A_coarse = []
    results_B_coarse = []
    
    for alpha in alpha_coarse:
        # Config A
        const_A = generate_apsk_custom(CONFIG_A, alpha)
        ser_A = simulate_ser(const_A, SNR_DB, NUM_SYMBOLS_COARSE)
        results_A_coarse.append(ser_A)
        
        # Config B
        const_B = generate_apsk_custom(CONFIG_B, alpha)
        ser_B = simulate_ser(const_B, SNR_DB, NUM_SYMBOLS_COARSE)
        results_B_coarse.append(ser_B)
        
        print(f"alpha={alpha:.1f} | SER_A={ser_A:.4e} | SER_B={ser_B:.4e}")
        
    # Find approx min to define fine sweep range
    min_idx_A = np.argmin(results_A_coarse)
    best_alpha_coarse_A = alpha_coarse[min_idx_A]
    
    min_idx_B = np.argmin(results_B_coarse)
    best_alpha_coarse_B = alpha_coarse[min_idx_B]
    
    print(f"Coarse Best: A approx {best_alpha_coarse_A:.1f}, B approx {best_alpha_coarse_B:.1f}")

    # --- Step 3: Fine Sweep ---
    print("\n--- Step 3: Fine Sweep (Around Best Coarse) ---")
    
    # Define a combined fine range to plot them together on one nice graph
    # Taking the range covering both best points +/- 0.1
    center_alpha = (best_alpha_coarse_A + best_alpha_coarse_B) / 2
    start_fine = max(0.2, center_alpha - 0.15)
    end_fine   = min(0.9, center_alpha + 0.15)
    
    alpha_fine = np.arange(start_fine, end_fine + 0.001, 0.02)
    
    results_A_fine = []
    results_B_fine = []
    
    for alpha in alpha_fine:
        # Config A
        const_A = generate_apsk_custom(CONFIG_A, alpha)
        ser_A = simulate_ser(const_A, SNR_DB, NUM_SYMBOLS_FINE)
        results_A_fine.append(ser_A)
        
        # Config B
        const_B = generate_apsk_custom(CONFIG_B, alpha)
        ser_B = simulate_ser(const_B, SNR_DB, NUM_SYMBOLS_FINE)
        results_B_fine.append(ser_B)
        
    # Find Absolute Best from Fine Sweep
    best_idx_A = np.argmin(results_A_fine)
    best_alpha_A = alpha_fine[best_idx_A]
    min_ser_A = results_A_fine[best_idx_A]
    
    best_idx_B = np.argmin(results_B_fine)
    best_alpha_B = alpha_fine[best_idx_B]
    min_ser_B = results_B_fine[best_idx_B]
    
    print(f"Fine Sweep Result:")
    print(f"Config A Best: alpha={best_alpha_A:.2f}, SER={min_ser_A:.5e}")
    print(f"Config B Best: alpha={best_alpha_B:.2f}, SER={min_ser_B:.5e}")
    
    # --- Step 4: 64QAM Comparison ---
    print("\n--- Step 4: 64QAM Comparison ---")
    const_qam = generate_64qam_constellation()
    ser_qam = simulate_ser(const_qam, SNR_DB, NUM_SYMBOLS_FINE)
    print(f"64-QAM SER={ser_qam:.5e}")
    
    # --- Step 5: Visualization ---
    print("\n--- Step 5: Generating Plots ---")
    
    # 1. Plot Best Constellation (Config A)
    const_best_A = generate_apsk_custom(CONFIG_A, best_alpha_A)
    plot_constellation(const_best_A, 
                      f'Best 64-APSK Config A (alpha={best_alpha_A:.2f})', 
                      'experiment_result_1_constellation_A.png')
    
    # 2. Plot Sweep Results
    plot_sweep_results(alpha_fine, results_A_fine, results_B_fine, 
                       (best_alpha_A, min_ser_A), (best_alpha_B, min_ser_B),
                       'experiment_result_2_sweep.png')
    
    # 3. Plot Comparison
    plot_comparison_bar(min_ser_A, ser_qam, f'64-APSK (A, a={best_alpha_A:.2f})',
                        'experiment_result_3_comparison.png')
    
    print("\n=== Experiment Completed Successfully ===")


if __name__ == "__main__":
    run_experiment()

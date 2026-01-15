import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

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



# ==========================================
# 5. Optimization Functions
# ==========================================

def generate_apsk_phased(ring_counts, alpha, phases):
    """
    Generates a 64-APSK constellation with specified ring counts, radius alpha, and phase offsets.
    
    Parameters:
        ring_counts (list): List of number of points per ring.
        alpha (float): Radius spacing factor.
        phases (list): Phase offsets for each ring in radians.
        
    Returns:
        points (np.array): Normalized complex array of constellation points.
    """
    """
    【責務】
    - リング構成、半径比(alpha)、および各リングの回転角(phases)を指定して64-APSK信号点配置を生成する。
    - `generate_apsk_custom` の拡張版であり、リングごとの位相回転（オフセット）を個別に制御可能。
    - 生成後の平均電力は1に正規化される。

    【手順書との対応】
    - 「各リングをどれだけ回転させるか」というパラメータ追加に対応。
    - 内側のリングに対して外側のリングをずらして配置する操作を実現する。
    """
    ring_counts = np.array(ring_counts)
    num_rings = len(ring_counts)
    
    # 1. Radii calculation (Same as before)
    gamma = np.array([1.0 + i * alpha for i in range(num_rings)])
    total_points = np.sum(ring_counts)
    energy_unscaled = np.sum(ring_counts * (gamma ** 2))
    R1 = np.sqrt(total_points / energy_unscaled)
    radii = gamma * R1
    
    points = []
    
    # 2. Point generation with phase offsets
    for r_idx in range(num_rings):
        n_points = ring_counts[r_idx]
        radius = radii[r_idx]
        # Use provided phase offset for this ring
        phase_offset = phases[r_idx] if r_idx < len(phases) else 0.0
        
        for i in range(n_points):
            theta = 2 * np.pi * i / n_points + phase_offset
            points.append(radius * np.exp(1j * theta))
            
    return np.array(points)

def calculate_dmin(points):
    """
    Calculates the minimum Euclidean distance between any pair of points in the constellation.
    """
    """
    【責務】
    - コンスタレーション点群内の全点ペア間のユークリッド距離を計算し、その最小値(d_min)を返す。
    - この値が大きいほど、雑音に対する耐性が強い（誤り率が低い）と期待される理論的指標。

    【手順書との対応】
    - 「理論的指標：最小ユークリッド距離の算出」に対応。
    - シミュレーションを行わずに配置の良し悪しを評価するコア関数。
    """
    # Convert complex points to (N, 2) real coordinates
    coords = np.column_stack((points.real, points.imag))
    
    # Calculate all pairwise distances
    dists = pdist(coords, 'euclidean')
    
    # Return the minimum distance
    # pdist returns a condensed distance matrix (1D array), so min() works directly
    if len(dists) == 0:
        return 0.0
    return np.min(dists)

def optimize_constellation(ring_counts):
    """
    Finds the optimal alpha and phase offsets to maximize d_min.
    Assumes the first ring phase is fixed at 0.
    """
    """
    【責務】
    - 指定されたリング構成に対して、d_min を最大化する最適な alpha と phases を自動探索する。
    - `scipy.optimize.minimize` を使用し、d_min の符号反転値を最小化（＝d_minを最大化）する。

    【手順書との対応】
    - 「自動探索：scipy.optimize などを使って…自動で探させる」に対応。
    - 手動スイープでは不可能な多次元パラメータ空間（alpha + 3つの回転角）の最適化を実現。

    【何をしているか】
    1. 最適化アルゴリズムに渡す「目的関数」を定義（d_minを計算してマイナスを付けて返す）。
    2. alpha（半径比）の探索範囲を [0.1, 1.5]、各リングの回転角の範囲を [0, 2π/点数] に設定。
    3. 数値計算（L-BFGS-B法）により、最も点同士が離れる「最強のalpha」と「最強の回転角」の組み合わせを算出。
    """
    num_rings = len(ring_counts)
    
    def objective(params):
        alpha = params[0]
        current_phases = [0.0] + list(params[1:])
        points = generate_apsk_phased(ring_counts, alpha, current_phases)
        d_min = calculate_dmin(points)
        return -d_min  # 最小化ツールなのでマイナスを付けて最大化させる
    
    # 初期値と探索範囲の設定
    x0 = [0.6] + [0.0] * (num_rings - 1)
    bounds = [(0.1, 1.5)] # alphaの範囲
    for count in ring_counts[1:]:
        bounds.append((0, 2 * np.pi / count)) # 回転角の範囲（1周期分）
        
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    best_params = result.x
    return best_params[0], [0.0] + list(best_params[1:]), -result.fun


def optimize_static_alpha(ring_counts):
    """
    回転角をすべて0に固定した状態で、d_minを最大化するalphaのみを探索する。
    （「回転の効果」を公平に検証するためのベースライン生成用）
    """
    def objective(params):
        alpha = params[0]
        phases = [0.0] * len(ring_counts)
        points = generate_apsk_phased(ring_counts, alpha, phases)
        d_min = calculate_dmin(points)
        return -d_min

    # alphaのみ探索
    res = minimize(objective, [0.6], bounds=[(0.1, 1.5)], method='L-BFGS-B')
    return res.x[0], -res.fun

def run_optimization_task():
    """
    【責務】
    - 複数のリング構成（候補）それぞれについて、以下の検証セットを全実行する：
      1. 回転ありの幾何学的最適化（Auto-Tuned）
      2. 回転なしでの幾何学的最適化（Baseline）
      3. 両者とQAMのSER比較シミュレーション（効果検証）
      4. 検証プロットの保存
    - 最後に全候補のランキングを表示する。

    【手順書との対応】
    - 「すべての構成について単一構成の最適化・回転なしとの比較検証を行う」という要望に対応。
    """
    print("=== Starting 64-APSK Multi-Config Optimization & Verification ===")
    
    # 検討したいリング構成のリスト
    candidate_configs = [
        [4, 12, 20, 28],  # 元々の案
        [8, 16, 20, 20],  # 以前の有力候補
        [1, 7, 19, 37],   # 中心に1点（蜂の巣状に近い）
        [6, 12, 18, 28],  # バランス型
        [4, 10, 20, 30],  # 外重点型
        [8, 12, 16, 28],  # 内重点型
    ]
    
    # 比較対象：64-QAM
    const_qam = generate_64qam_constellation()
    dmin_qam = calculate_dmin(const_qam)
    SNR_DB = 18.0
    print(f"Reference: 64-QAM (d_min={dmin_qam:.4f}) calculating SER...")
    ser_qam = simulate_ser(const_qam, SNR_DB, 100000)
    print(f"Reference: 64-QAM SER={ser_qam:.5e}")
    
    results_summary = []

    for config in candidate_configs:
        config_str = str(config).replace(' ', '')
        print(f"\n" + "="*50)
        print(f"Testing Config: {config}")
        print("="*50)
        
        # 1. 回転ありの最適化 (Optimized)
        #    - d_minが最大になる alpha と phases を探す
        alpha_opt, phases_opt, dmin_opt = optimize_constellation(config)
        
        # 2. 回転なしの最適化 (Baseline)
        #    - 回転角0固定で、ベストな alpha だけ探す（公平な比較のため）
        alpha_base, dmin_base = optimize_static_alpha(config)
        phases_base = [0.0] * len(config)
        
        print(f"  [Geometry] Baseline(Rot=0) d_min: {dmin_base:.4f} (alpha={alpha_base:.3f})")
        print(f"  [Geometry] Optimized(Rot=*) d_min: {dmin_opt:.4f} (alpha={alpha_opt:.3f})")
        print(f"  -> d_min Improvement: {((dmin_opt/dmin_base)-1)*100:.1f}%")

        # 3. SERシミュレーション (回転の効果検証)
        points_base = generate_apsk_phased(config, alpha_base, phases_base)
        points_opt  = generate_apsk_phased(config, alpha_opt, phases_opt)
        
        ser_base = simulate_ser(points_base, SNR_DB, 100000)
        ser_opt  = simulate_ser(points_opt, SNR_DB, 100000)
        
        print(f"  [Simulation] Baseline SER: {ser_base:.5e}")
        print(f"  [Simulation] Optimized SER: {ser_opt:.5e}")
        print(f"  -> SER Reduction: {(1 - ser_opt/ser_base)*100:.1f}%")
        
        # 4. 個別比較プロットの作成 (Verification Plot)
        labels = ['Baseline', 'Optimized', '64-QAM']
        values = [ser_base, ser_opt, ser_qam]
        
        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, values, color=['gray', 'blue', 'red'])
        plt.title(f'Effect of Rotation Optimization\nConfig {config} @ SNR={SNR_DB}dB')
        plt.ylabel('Symbol Error Rate (SER)')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2e}', ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        filename = f'verify_config_{config_str}.png'
        plt.savefig(filename)
        plt.close()
        print(f"  -> Saved verification plot: {filename}")
        
        # 結果を保存
        results_summary.append({
            'config': config, 
            'alpha': alpha_opt, 'phases': phases_opt,
            'dmin': dmin_opt, 'ser': ser_opt
        })

    # --- 最終結果ランキングの表示 ---
    print("\n" + "="*60)
    print(f"{'Ranking':<3} | {'Config':<20} | {'d_min':<8} | {'SER':<10} | {'vs QAM'}")
    print("-" * 60)
    
    # d_min が大きい順に並び替え（幾何学的優位性順）
    results_summary.sort(key=lambda x: x['dmin'], reverse=True)
    
    for rank, res in enumerate(results_summary, 1):
        status = "WIN" if res['dmin'] > dmin_qam else "LOSE"
        if res['ser'] < ser_qam: 
            status += " (SER OK)"
        else:
            status += " (SER NG)"
            
        print(f"{rank:<3} | {str(res['config']):<20} | {res['dmin']:.4f} | {res['ser']:.2e} | {status}")
    print("="*60)
    print(f"(Reference: 64-QAM d_min = {dmin_qam:.4f}, SER = {ser_qam:.2e})")

    # 最強構成のコンスタレーションを保存
    best = results_summary[0]
    p_best = generate_apsk_phased(best['config'], best['alpha'], best['phases'])
    plot_constellation(p_best, f"Best Config {best['config']} (d_min={best['dmin']:.3f})", "best_overall_constellation.png")

if __name__ == "__main__":
    run_optimization_task()

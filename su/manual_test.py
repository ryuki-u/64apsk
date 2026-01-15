import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import itertools
import os

"""
=============================================================================
64-APSK Constellation Multi-Config Brute-Force Optimizer (Catalog Generator)
=============================================================================

【プログラムの目的】
本プログラムは、64-APSK信号点配置において、通信品質の要となる「最小ユークリッド距離(d_min)」
を最大化する最適な半径比(alpha)と、各リングの回転角(phases)を特定するための全探索ツールです。

【背景と設計思想：なぜ「自動最適化」ではなく「全探索」なのか】
1. 非凸最適化の回避: 
   d_minの最適化問題は、多次元かつ局所解（ローカルミニマム）が非常に多い性質を持ちます。
   勾配法などのアルゴリズムでは、初期値(phases=0)から動けなくなる「スタック」が発生
   しやすいため、本コードでは人間が指定した範囲を「しらみつぶし」に計算します。
   
2. ジグザグ配置（蜂の巣状配置）の発見:
   隣接するリング同士を特定の角度で回転させることで、点と点の隙間に外側の点を滑り込ませる
   「ジグザグ配置」は、計算上の微細な黄金比によって成り立っています。
   全1,000通りの位相組み合わせを検証することで、この「カチッとはまる」瞬間を確実に捕捉します。

【処理フロー（3段階スイープ）】
Step A [土台の選定]: 
    回転角を0に固定し、半径比(alpha)をスイープして、大まかなリング間隔のベストを探します。
Step B [ジグザグ化の検証]: 
    Step Aのベストalphaにおいて、第2・3・4リングを独立に回転（10段階ずつ、計1,000通り）
    させ、d_minが跳ね上がる「回転の黄金比」を探索します。
Step C [カタログ化]: 
    各リング構成の「ポテンシャル（回転による改善率: Gain）」を算出し、ランキング形式で
    記録。上位のパラメータを「optimized_catalog.txt」に保存します。

【成果物】
- Ranking Table: どのリング構成が最もポテンシャルが高いかの比較表。
- Sensitivity Plots: 位相の変化に対してd_minがどれだけ敏感に反応するかの可視化。
- Constellation Plots: 理論上最強となった信号点配置図。

提示されたコードは、各リング構成（Standard, Uniform, InnerHeavy, Honeycomb）に対して、
「感度グラフ」と「最適化された配置図」の2種類、計8枚の画像を出力します。
具体的にどのような画像が、どの名前で保存されるのかを以下に整理します。

1. 配置図：catalog_[構成名]_best_const.png
   各リング構成において、最も d_min が高くなった瞬間の信号点の並びを可視化した画像です。
   - 見た目: 青い点（信号点）が同心円状に並んでいます。
   - 補助線: 各リングの半径を示す灰色の点線の円が描かれ、中心には十字の基準線があります。
   - タイトル: 構成名と、その時の d_min の値が表示されます。
   - ファイル名の例: catalog_Standard_best_const.png, catalog_Uniform_best_const.png
     に並んで隙間を埋めているかを確認するために出力します。

   【各構成の配列（リング内点数）の定義】
   - Standard   : [4, 12, 20, 28] (等間隔な基準配置)
   - Uniform    : [8, 16, 20, 20] (密度が均一な配置)
   - InnerHeavy : [8, 12, 16, 28] (内側の密度を高め、外側のd_minを稼ぐ配置)
   - Honeycomb  : [1, 7, 13, 19, 24] (中心1点、理想的な六方格子に近い配置)
   - BalancedV2 : [8, 14, 18, 24] (各リングの密度を緩やかに変化させた調整型)

2. 感度グラフ：catalog_[構成名]_sensitivity.png
   総当たり（全1,000通り）で計算した回転角の組み合わせを、成績の良い順（d_min が大きい順）に
   並べ替えてプロットしたグラフです。
   - 横軸: 試行インデックス（0から999まで、成績順）。
   - 縦軸: d_min の値。
   - 見た目: 左側（成績が良い組み合わせ）から右側に向かって、値が下がっていく曲線になります。
   - 設計の狙い: グラフの左端の傾斜を見ることで、その配置が少しの回転ズレで性能が落ちる
     シビアなものか、それとも安定しているかを判断するために出力します。

=============================================================================
"""

# ==========================================
# 1. Base Logic (Generation & Metrics)
# ==========================================

def generate_apsk_phased(ring_counts, alpha, phases):
    """
    Generates a 64-APSK constellation with specified parameters.
    
    Args:
        ring_counts (list): Number of points per ring.
        alpha (float): Radius spacing factor. Radii = R1 * [1, 1+a, 1+2a...]
        phases (list): Phase offset (radians) for each ring.
        
    Returns:
        points (np.array): Normalized complex points (Avg Power = 1).
    """
    ring_counts = np.array(ring_counts)
    num_rings = len(ring_counts)
    phases = np.array(phases)
    
    # Pad phases if shorter than rings
    if len(phases) < num_rings:
        phases = np.pad(phases, (0, num_rings - len(phases)), 'constant')
        
    # 1. Radii Calculation
    # gamma = [1, 1+alpha, 1+2alpha, ...]
    gamma = np.array([1.0 + i * alpha for i in range(num_rings)])
    
    # Normalize Energy
    # Total Energy = sum(N_i * R_i^2)
    # We want Total Energy / Total Points = 1
    total_points = np.sum(ring_counts)
    energy_unscaled = np.sum(ring_counts * (gamma ** 2))
    R1 = np.sqrt(total_points / energy_unscaled)
    radii = gamma * R1
    
    points = []
    
    # 2. Point Generation
    for r_idx in range(num_rings):
        n = int(ring_counts[r_idx])
        r = radii[r_idx]
        phi = phases[r_idx]
        
        # Generate N points distributed evenly, starting at angle phi
        k = np.arange(n)
        theta = 2 * np.pi * k / n + phi
        
        ring_points = r * np.exp(1j * theta)
        points.append(ring_points)
            
    return np.concatenate(points)

def calculate_dmin(points):
    """
    Calculates the minimum Euclidean distance (d_min) in the constellation.
    """
    if len(points) < 2:
        return 0.0
    
    coords = np.column_stack((points.real, points.imag))
    # pdist computes pairwise distances between observations in n-dimensional space
    dists = pdist(coords, 'euclidean')
    return np.min(dists)

def plot_constellation(points, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(points.real, points.imag, c='blue', marker='o', s=30)
    
    # Draw loops for visual reference
    rs = np.unique(np.abs(points))
    for r in rs:
        circle = plt.Circle((0,0), r, fill=False, linestyle=':', color='gray', alpha=0.5)
        plt.gca().add_artist(circle)
        
    plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

# ==========================================
# 2. Simulation Functions (SER)
# ==========================================

def simulate_ser(constellation, snr_db, num_symbols=100000):
    """
    Simulates Symbol Error Rate (SER) for a given constellation at a specific SNR.
    Uses Nearest Neighbor (ML) decoding.
    """
    M = len(constellation)
    
    # 1. Generate random Indices
    tx_indices = np.random.randint(0, M, num_symbols)
    tx_symbols = constellation[tx_indices]
    
    # 2. Add AWGN
    snr_linear = 10**(snr_db / 10.0)
    n0 = 1.0 / snr_linear
    noise_std = np.sqrt(n0 / 2)
    
    noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = tx_symbols + noise
    
    # 3. Demodulate (Nearest Neighbor)
    rx_coords = np.column_stack((rx_symbols.real, rx_symbols.imag))
    const_coords = np.column_stack((constellation.real, constellation.imag))
    
    from scipy.spatial import distance
    dists = distance.cdist(rx_coords, const_coords, 'euclidean')
    estimated_indices = np.argmin(dists, axis=1)
    
    # 4. Calculate SER
    errors = np.sum(tx_indices != estimated_indices)
    return errors / num_symbols

# ==========================================
# 3. Brute-Force Sweep Engines
# ==========================================

def sweep_alpha(ring_counts, alpha_range, fixed_phases=None):
    """
    Step A engine: Sweep alpha while keeping phases fixed.
    Also fulfills the 'reverse' condition (Optimizing alpha for a given rotation).
    """
    if fixed_phases is None:
        fixed_phases = [0.0] * len(ring_counts)
        
    results = []
    best_dmin = -1.0
    best_alpha = None
    
    for a in alpha_range:
        pts = generate_apsk_phased(ring_counts, a, fixed_phases)
        d = calculate_dmin(pts)
        results.append((a, d))
        
        if d > best_dmin:
            best_dmin = d
            best_alpha = a
            
    return best_alpha, best_dmin, results

def sweep_phases_bruteforce(ring_counts, fixed_alpha, divisions=10):
    """
    Step B engine: Sweep phase rotations while keeping alpha fixed.
    Explores all combinations of discretized rotations for rings 2..M (Ring 1 fixed at 0).
    """
    num_rings = len(ring_counts)
    
    # Generate phase options for each ring
    # Ring 0: Fixed at 0.0 (Global rotation doesn't change d_min)
    phase_options = [[0.0]]
    
    # Rings 1..M-1: Discretize [0, 2pi/N) into 'divisions' steps
    for r_idx in range(1, num_rings):
        n = ring_counts[r_idx]
        period = 2 * np.pi / n
        step = period / divisions
        # Only go up to period-step (endpoint=False)
        angles = np.arange(0, period - 0.000001, step)
        phase_options.append(angles)
        
    # Create iterator for all combinations (Cartesian Product)
    # WARNING: Complexity is divisions^(num_rings-1). 
    # For 4 rings, divisions=10, 10^3=1000. Very fast.
    # For 5 rings, divisions=10, 10^4=10000. Still fast.
    all_combinations = itertools.product(*phase_options)
    
    results = []
    best_dmin = -1.0
    best_phases = None
    
    for phases in all_combinations:
        phs_list = list(phases)
        pts = generate_apsk_phased(ring_counts, fixed_alpha, phs_list)
        d = calculate_dmin(pts)
        results.append((phs_list, d))
        
        if d > best_dmin:
            best_dmin = d
            best_phases = phs_list
            
    return best_phases, best_dmin, results


# ==========================================
# 3. Main Cataloging Routine
# ==========================================

def run_catalog_creation():
    print("=== 64-APSK Brute Force Catalog Creation ===")
    
    # 成果物の保存ディレクトリ
    output_dir = "data/manual_su"
    os.makedirs(output_dir, exist_ok=True)
    
    # 網羅的探索リストの定義（全探索に近い広範囲な検証）
    raw_configs = [
        # --- 4 Rings ---
        [4, 12, 20, 28], [4, 10, 20, 30], [4, 8, 12, 40],
        [6, 12, 18, 28], [6, 10, 15, 33], [6, 14, 20, 24],
        [8, 12, 16, 28], [8, 14, 18, 24],[8, 14, 20, 22], [8, 16, 20, 20], [8, 12, 20, 24],
        [10, 14, 18, 22], [12, 16, 18, 18],
        # --- 5 Rings ---
        [1, 7, 13, 19, 24], [1, 6, 12, 18, 27], [1, 5, 10, 15, 33],
        [4, 8, 12, 16, 24], [4, 6, 10, 14, 30],
        [6, 10, 14, 16, 18], [8, 10, 12, 16, 18],
        # --- 3 Rings (Reference) ---
        [12, 20, 32], [16, 24, 24]
    ]

    # 辞書形式に変換（「Conf_8-14-18-24」のような形式で自動命名）
    target_configs = {f"Conf_{'-'.join(map(str, c))}": c for c in raw_configs}
    
    # Protocol Parameters
    # Step A: Wide Alpha Search
    alpha_range_step_a = np.arange(0.4, 1.25, 0.05) 
    
    # Step B: Phase Search Precision
    phase_divisions = 10
    SNR_DB = 18.0
    
    ranking_data = []

    for name, rings in target_configs.items():
        print(f"\n>> Processing Config: {name} {rings}")
        
        # --- Step A: Find Best Alpha (Base) ---
        best_alpha, base_dmin, res_alpha = sweep_alpha(rings, alpha_range_step_a, fixed_phases=None)
        
        # --- Step B: Sweep Phases at Best Alpha ---
        best_phases, best_dmin, res_phases = sweep_phases_bruteforce(rings, best_alpha, divisions=phase_divisions)
        
        # --- Step C: Calculate SER for the best rotation ---
        pts_best = generate_apsk_phased(rings, best_alpha, best_phases)
        ser_val = simulate_ser(pts_best, SNR_DB, num_symbols=100000)
        
        # Calculate Improvement
        gain = (best_dmin / base_dmin - 1) * 100 if base_dmin > 0 else 0
        print(f"   [Result] d_min={best_dmin:.4f} (Gain +{gain:.1f}%) | SER={ser_val:.2e} (@{SNR_DB}dB)")
        
        # Store for ranking
        ranking_data.append({
            'name': name,
            'rings': rings,
            'best_alpha': best_alpha,
            'best_phases': best_phases,
            'dmin': best_dmin,
            'ser': ser_val,
            'base_dmin': base_dmin,
            'gain': gain,
            'all_phase_results': res_phases
        })
        
        # Plot Sensitivity (d_min vs Sorted Combination Index)
        sorted_dmins = sorted([r[1] for r in res_phases], reverse=True)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_dmins)
        plt.title(f"{name} Phase Sensitivity\n(Alpha={best_alpha:.2f}, {len(sorted_dmins)} configs)")
        plt.xlabel("Configuration Index (Sorted by Performance)")
        plt.ylabel("d_min")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"catalog_{name}_sensitivity.png"))
        plt.close()
        
        # Plot Best Constellation
        pts_best = generate_apsk_phased(rings, best_alpha, best_phases)
        plot_constellation(pts_best, 
                          f"{name} Optimized (d_min={best_dmin:.4f})", 
                          os.path.join(output_dir, f"catalog_{name}_best_const.png"))

    # --- Final Output: Ranking Table ---
    print("\n" + "="*95)
    print(f" FINAL RANKING (Sorted by Optimized d_min)")
    print("="*95)
    print(f"{'Rank':<4} | {'Name':<18} | {'Alpha':<6} | {'d_min':<8} | {'SER (@18dB)':<10} | {'Gain':<6} | {'Phases (deg)'}")
    print("-" * 95)
    
    ranking_data.sort(key=lambda x: x['dmin'], reverse=True)
    
    for i, data in enumerate(ranking_data, 1):
        phases_deg = [round(np.degrees(p), 1) for p in data['best_phases']]
        print(f"{i:<4} | {data['name']:<18} | {data['best_alpha']:<6.2f} | {data['dmin']:<8.4f} | {data['ser']:<10.2e} | +{data['gain']:<4.1f}% | {phases_deg}")
    
    print("="*95)
    
    # Save Catalog as Markdown Table
    with open("optimized_catalog.txt", "w") as f:
        f.write("# 64-APSK Optimized Configuration Catalog\n\n")
        f.write("| Ring Configuration | d_min | SER (@18dB) | alpha | Gain | Phases (deg) |\n")
        f.write("|:------------------:|:-----:|:-----------:|:-----:|:----:|:------------:|\n")
        for data in ranking_data:
            phases_deg = [round(np.degrees(p), 1) for p in data['best_phases']]
            f.write(f"| {str(data['rings']):<18} | {data['dmin']:.5f} | {data['ser']:.2e} | {data['best_alpha']:.2f} | +{data['gain']:>4.1f}% | {phases_deg} |\n")
            
    print("Catalog saved to 'optimized_catalog.txt' as a Markdown table.")

if __name__ == "__main__":
    run_catalog_creation()

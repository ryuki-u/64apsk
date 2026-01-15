=== Starting 64-APSK Multi-Config Optimization & Verification ===
Reference: 64-QAM (d_min=0.3086) calculating SER...
Reference: 64-QAM SER=1.39980e-01

==================================================
Testing Config: [4, 12, 20, 28]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.2791 (alpha=1.500)
  [Geometry] Optimized(Rot=*) d_min: 0.2791 (alpha=1.500)
  -> d_min Improvement: 0.0%
  [Simulation] Baseline SER: 1.44390e-01
  [Simulation] Optimized SER: 1.43670e-01
  -> SER Reduction: 0.5%
  -> Saved verification plot: verify_config_[4,12,20,28].png

==================================================
Testing Config: [8, 16, 20, 20]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.2837 (alpha=0.640)
  [Geometry] Optimized(Rot=*) d_min: 0.2837 (alpha=0.640)
  -> d_min Improvement: 0.0%
  [Simulation] Baseline SER: 1.43280e-01
  [Simulation] Optimized SER: 1.41390e-01
  -> SER Reduction: 1.3%
  -> Saved verification plot: verify_config_[8,16,20,20].png

==================================================
Testing Config: [1, 7, 19, 37]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.1948 (alpha=1.500)
  [Geometry] Optimized(Rot=*) d_min: 0.1948 (alpha=1.500)
  -> d_min Improvement: 0.0%
  [Simulation] Baseline SER: 2.33190e-01
  [Simulation] Optimized SER: 2.33000e-01
  -> SER Reduction: 0.1%
  -> Saved verification plot: verify_config_[1,7,19,37].png

==================================================
Testing Config: [6, 12, 18, 28]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.2798 (alpha=1.155)
  [Geometry] Optimized(Rot=*) d_min: 0.2798 (alpha=1.155)
  -> d_min Improvement: -0.0%
  [Simulation] Baseline SER: 1.40350e-01
  [Simulation] Optimized SER: 1.42510e-01
  -> SER Reduction: -1.5%
  -> Saved verification plot: verify_config_[6,12,18,28].png

==================================================
Testing Config: [4, 10, 20, 30]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.2557 (alpha=1.500)
  [Geometry] Optimized(Rot=*) d_min: 0.2557 (alpha=1.500)
  -> d_min Improvement: 0.0%
  [Simulation] Baseline SER: 1.59100e-01
  [Simulation] Optimized SER: 1.56810e-01
  -> SER Reduction: 1.4%
  -> Saved verification plot: verify_config_[4,10,20,30].png

==================================================
Testing Config: [8, 12, 16, 28]
==================================================
  [Geometry] Baseline(Rot=0) d_min: 0.2785 (alpha=0.806)
  [Geometry] Optimized(Rot=*) d_min: 0.2785 (alpha=0.806)
  -> d_min Improvement: 0.0%
  [Simulation] Baseline SER: 1.46070e-01
  [Simulation] Optimized SER: 1.45710e-01
  -> SER Reduction: 0.2%
  -> Saved verification plot: verify_config_[8,12,16,28].png

============================================================
Ranking | Config               | d_min    | SER        | vs QAM
------------------------------------------------------------
1   | [8, 16, 20, 20]      | 0.2837 | 1.41e-01 | LOSE (SER NG)
2   | [6, 12, 18, 28]      | 0.2798 | 1.43e-01 | LOSE (SER NG)
3   | [4, 12, 20, 28]      | 0.2791 | 1.44e-01 | LOSE (SER NG)
4   | [8, 12, 16, 28]      | 0.2785 | 1.46e-01 | LOSE (SER NG)
5   | [4, 10, 20, 30]      | 0.2557 | 1.57e-01 | LOSE (SER NG)
6   | [1, 7, 19, 37]       | 0.1948 | 2.33e-01 | LOSE (SER NG)
============================================================
(Reference: 64-QAM d_min = 0.3086, SER = 1.40e-01)
Saved plot: best_overall_constellation.png
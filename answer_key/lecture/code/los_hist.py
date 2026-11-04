"""
los_hist.py — 재원일수(los_days) 분포 확인: 히스토그램과 기술통계 (검정은 하지 않음)
사용법: python code/los_hist.py [입력 CSV] [출력 PNG]
  기본 입력: data/appendicitis_synthetic.csv
  기본 출력: results/figures/los_hist.png
의존성: pandas, matplotlib
※ 강의용 합성데이터. 실제 환자 아님.

제외 기준은 Table 1과 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸)
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/appendicitis_synthetic.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/figures/los_hist.png"
TREATS = ["LA", "OA", "NOM"]
NAMES = {"LA": "Laparoscopic (LA)", "OA": "Open (OA)", "NOM": "Non-operative (NOM)"}

df = pd.read_csv(src, encoding="utf-8-sig", dtype={"sex": str})
blank = df.sex.isna() | (df.sex.str.strip() == "")
d = df[(df.age <= 120) & (df.los_days >= 0) & ~blank].copy()
print(f"분석 대상 {len(d):,}명 (원자료 {len(df):,}명, 제외 {len(df) - len(d)}명)\n")

# 기술통계 (탭 구분)
cols = ["group", "n", "mean", "sd", "variance", "median", "q1", "q3", "min", "max", "skewness", "n_ge_14", "n_ge_30"]
print("\t".join(cols))
for g, s in [("ALL", d.los_days)] + [(t, d[d.treatment == t].los_days) for t in TREATS]:
    v = [g, len(s), f"{s.mean():.1f}", f"{s.std():.1f}", f"{s.var():.1f}", f"{s.median():.1f}",
         f"{s.quantile(.25):.1f}", f"{s.quantile(.75):.1f}", s.min(), s.max(), f"{s.skew():.2f}",
         int((s >= 14).sum()), int((s >= 30).sum())]
    print("\t".join(map(str, v)))

# 그림: 왼쪽 전체 분포, 오른쪽 치료방법별 분포(밀도)
os.makedirs(os.path.dirname(dst), exist_ok=True)
bins = np.arange(0, d.los_days.max() + 2) - 0.5
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

ax[0].hist(d.los_days, bins=bins, color="#1E6BB5", edgecolor="white")
ax[0].axvline(d.los_days.mean(), color="#C0392B", ls="--", label=f"Mean {d.los_days.mean():.1f}")
ax[0].axvline(d.los_days.median(), color="#1B2A4A", ls="-", label=f"Median {d.los_days.median():.1f}")
ax[0].set(title=f"Length of stay, all patients (n = {len(d):,})", xlabel="Length of stay (days)", ylabel="Number of patients")
ax[0].legend(frameon=False)

colors = {"LA": "#1E6BB5", "OA": "#C0392B", "NOM": "#2E8B57"}
for t in TREATS:
    s = d[d.treatment == t].los_days
    ax[1].hist(s, bins=bins, density=True, histtype="step", lw=1.8, color=colors[t],
               label=f"{NAMES[t]}, n = {len(s):,}")
ax[1].set(title="Length of stay by treatment", xlabel="Length of stay (days)", ylabel="Proportion of patients")
ax[1].legend(frameon=False)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Synthetic data for teaching; not real patients.", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(dst, dpi=150)
print(f"\nsaved {dst}")

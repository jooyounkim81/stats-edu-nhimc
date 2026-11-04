"""
trend.py — 재원일수(los_days)의 연도별 추세: 중앙값 그래프와 log 선형회귀(보정 전후)
사용법: python code/trend.py [입력 CSV] [그림 PNG] [결과 CSV]
  기본 입력: data/appendicitis_synthetic.csv
  기본 출력: results/figures/los_trend.png, results/trend.csv
의존성: pandas, numpy, statsmodels, matplotlib
※ 강의용 합성데이터. 실제 환자 아님.

제외 기준: Table 1과 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸). 재원일수는 모두 1일 이상이라 log를 그대로 쓴다.
모형 1 (보정 전): log(los_days) ~ year
모형 2 (보정 후): log(los_days) ~ year + treatment + complicated + age_group + comorbidity
  기준군: treatment = LA, age_group = 18-64. year는 연속형.
연간 변화율(%) = (exp(year 계수) − 1) × 100. 기하평균 재원일수가 1년마다 몇 % 변하는지를 뜻한다.
95% 신뢰구간: OLS 표준오차와 이분산에 강건한 HC3 표준오차를 함께 보고한다.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = sys.argv[1] if len(sys.argv) > 1 else "data/appendicitis_synthetic.csv"
fig_path = sys.argv[2] if len(sys.argv) > 2 else "results/figures/los_trend.png"
csv_path = sys.argv[3] if len(sys.argv) > 3 else "results/trend.csv"
TREATS = ["LA", "OA", "NOM"]

df = pd.read_csv(src, encoding="utf-8-sig", dtype={"sex": str})
blank = df.sex.isna() | (df.sex.str.strip() == "")
d = df[(df.age <= 120) & (df.los_days >= 0) & ~blank].copy()
assert (d.los_days > 0).all(), "재원일수 0 이하가 있어 log를 쓸 수 없음"
d["treatment"] = pd.Categorical(d.treatment, TREATS)
d["age_group"] = pd.Categorical(d.age_group, ["18-64", "<18", ">=65"])
print(f"분석 대상 {len(d):,}명 (원자료 {len(df):,}명, 제외 {len(df) - len(d)}명)\n")

# 연도별 요약: 중앙값, 치료방법별 중앙값, 환자 구성
g = d.groupby("year")
yr = pd.DataFrame({
    "n": g.size(),
    "median_all": g.los_days.median(),
    **{f"median_{t}": d[d.treatment == t].groupby("year").los_days.median() for t in TREATS},
    "OA_pct": g.treatment.apply(lambda s: (s == "OA").mean() * 100),
    "complicated_pct": g.complicated.mean() * 100,
    "ge65_pct": g.age_group.apply(lambda s: (s == ">=65").mean() * 100),
}).round(1)
print("[연도별 요약]")
print(yr.reset_index().to_csv(index=False, sep="\t"), end="")

# 회귀: 보정 전후
models = {
    "보정 전": "np.log(los_days) ~ year",
    "보정 후": "np.log(los_days) ~ year + treatment + complicated + age_group + comorbidity",
}
rows = []
for name, f in models.items():
    for se_type in ["OLS", "HC3"]:
        m = smf.ols(f, d).fit() if se_type == "OLS" else smf.ols(f, d).fit(cov_type="HC3")
        b = m.params["year"]
        lo, hi = m.conf_int().loc["year"]
        p = m.pvalues["year"]
        rows.append({"모형": name, "표준오차": se_type, "n": int(m.nobs),
                     "연간 변화율(%)": round((np.exp(b) - 1) * 100, 1),
                     "95% CI 하한": round((np.exp(lo) - 1) * 100, 1),
                     "95% CI 상한": round((np.exp(hi) - 1) * 100, 1),
                     "p값": "<0.001" if p < 0.001 else f"{p:.3f}",
                     "R2": round(m.rsquared, 3)})
res = pd.DataFrame(rows)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
res.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("\n[연간 변화율]")
print(res.to_csv(index=False, sep="\t"), end="")

# 그림: 왼쪽 연도별 중앙값(전체·치료방법별), 오른쪽 연도별 개복 수술 비율
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
years = yr.index.values
colors = {"LA": "#1E6BB5", "OA": "#C0392B", "NOM": "#2E8B57"}
names = {"LA": "Laparoscopic (LA)", "OA": "Open (OA)", "NOM": "Non-operative (NOM)"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax[0].plot(years, yr.median_all, "-o", color="#1B2A4A", lw=2.5, label="All patients")
for t in TREATS:
    ax[0].plot(years, yr[f"median_{t}"], "--s", color=colors[t], lw=1.3, ms=4, label=names[t])
ax[0].set(title="Median length of stay by year", xlabel="Year of diagnosis", ylabel="Median length of stay (days)")
ax[0].set_ylim(0, None)
ax[0].legend(frameon=False, fontsize=9)
ax[1].bar(years, yr.OA_pct, color=colors["OA"], alpha=0.8)
ax[1].set(title="Share of open surgery (OA) by year", xlabel="Year of diagnosis", ylabel="Patients treated with OA (%)")
for a in ax:
    a.set_xticks(years, [str(y) for y in years], rotation=45)
    a.spines[["top", "right"]].set_visible(False)
crude = res[(res.모형 == "보정 전") & (res.표준오차 == "OLS")].iloc[0]
adj = res[(res.모형 == "보정 후") & (res.표준오차 == "OLS")].iloc[0]
fig.text(0.01, 0.01, f"Annual change in geometric mean LOS: crude {crude['연간 변화율(%)']:+.1f}%, "
         f"adjusted {adj['연간 변화율(%)']:+.1f}% (treatment, complicated, age group, comorbidity). "
         "Synthetic data for teaching; not real patients.", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(fig_path, dpi=150)
print(f"\nsaved {csv_path}, {fig_path}")

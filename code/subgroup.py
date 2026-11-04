"""
subgroup.py — 개복(OA) 대 복강경(LA) 합병증 승산비의 하위군 분석(연령·성별)과 상호작용 검정
사용법: python code/subgroup.py [입력 CSV] [그림 PNG] [결과 CSV]
  기본 입력: data/appendicitis_synthetic.csv
  기본 출력: results/figures/subgroup_forest.png, results/subgroup.csv
의존성: pandas, numpy, statsmodels, matplotlib
※ 강의용 합성데이터. 실제 환자 아님.

제외 기준: Table 1과 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸)
하위군 모형: complication ~ treatment + complicated + comorbidity (각 하위군 안에서 적합)
상호작용 모형: complication ~ treatment * old + complicated + comorbidity
              complication ~ treatment * male + complicated + comorbidity
  old = 65세 이상(age_group ">=65") 1/0, male = 남성 1/0
  treatment는 LA(기준)·OA·NOM 세 범주를 모두 넣고, OA 대 LA 항만 보고한다(02_analysis.py와 같은 방식).
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
fig_path = sys.argv[2] if len(sys.argv) > 2 else "results/figures/subgroup_forest.png"
csv_path = sys.argv[3] if len(sys.argv) > 3 else "results/subgroup.csv"
TERM = "treatment[T.OA]"

df = pd.read_csv(src, encoding="utf-8-sig", dtype={"sex": str})
blank = df.sex.isna() | (df.sex.str.strip() == "")
d = df[(df.age <= 120) & (df.los_days >= 0) & ~blank].copy()
d["sex"] = d.sex.str.strip()
d["old"] = (d.age_group == ">=65").astype(int)
d["male"] = (d.sex == "M").astype(int)
d["treatment"] = pd.Categorical(d.treatment, ["LA", "OA", "NOM"])
print(f"분석 대상 {len(d):,}명 (원자료 {len(df):,}명, 제외 {len(df) - len(d)}명)\n")


def fit_or(data, formula="complication ~ treatment + complicated + comorbidity"):
    m = smf.logit(formula, data).fit(disp=0)
    lo, hi = np.exp(m.conf_int().loc[TERM])
    return np.exp(m.params[TERM]), lo, hi, m.pvalues[TERM]


def events(data, t):
    s = data[data.treatment == t]
    return f"{int(s.complication.sum())}/{len(s)}"


# 하위군별 승산비
subs = [("Overall", "전체", None, d.index == d.index),
        ("Age", "65세 미만", "Age < 65", d.old == 0), ("Age", "65세 이상", "Age >= 65", d.old == 1),
        ("Sex", "여성", "Female", d.male == 0), ("Sex", "남성", "Male", d.male == 1)]
rows = []
for grp, ko, en, mask in subs:
    s = d[mask]
    orv, lo, hi, p = fit_or(s)
    rows.append({"구분": grp, "하위군": ko, "label": en or "All patients", "n": len(s),
                 "OA 합병증/인원": events(s, "OA"), "LA 합병증/인원": events(s, "LA"),
                 "승산비": round(orv, 2), "95% CI 하한": round(lo, 2), "95% CI 상한": round(hi, 2), "p값": p})

# 상호작용 검정
p_int = {}
for var, grp in [("old", "Age"), ("male", "Sex")]:
    m = smf.logit(f"complication ~ treatment * {var} + complicated + comorbidity", d).fit(disp=0)
    term = f"{TERM}:{var}"
    p_int[grp] = m.pvalues[term]
    print(f"[상호작용] treatment(OA) × {var}: 승산비 비 {np.exp(m.params[term]):.2f}, p = {m.pvalues[term]:.3f}")

t = pd.DataFrame(rows)
t["상호작용 p값"] = t.구분.map(lambda g: p_int.get(g, np.nan))
out = t.drop(columns="label").copy()
fmt = lambda p: "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
out["p값"] = out["p값"].map(fmt)
out["상호작용 p값"] = out["상호작용 p값"].map(fmt)
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print()
print(out.to_csv(index=False, sep="\t"), end="")

# 포리스트 그림 (그림 글자는 영어)
from matplotlib.ticker import NullLocator
HEAD = {"Age": "Age group", "Sex": "Sex"}
items, prev = [], None          # (y, 라벨, 굵게 여부, 결과 행 또는 None)
y = 0
for r in rows:
    if r["구분"] in HEAD and r["구분"] != prev:
        y -= 1
        items.append((y, f"{HEAD[r['구분']]} (P for interaction = {p_int[r['구분']]:.2f})", True, None))
    prev = r["구분"]
    y -= 1
    indent = "" if r["구분"] == "Overall" else "    "
    items.append((y, f"{indent}{r['label']} (n = {r['n']:,})", r["구분"] == "Overall", r))

os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 4.4))
for yy, _, _, r in items:
    if r is None:
        continue
    c = "#1B2A4A" if r["구분"] == "Overall" else "#1E6BB5"
    ax.plot([r["95% CI 하한"], r["95% CI 상한"]], [yy, yy], color=c, lw=2)
    ax.plot(r["승산비"], yy, "D" if r["구분"] == "Overall" else "s", color=c, ms=8)
    ax.text(1.03, yy, f"{r['승산비']:.2f} ({r['95% CI 하한']:.2f}–{r['95% CI 상한']:.2f})",
            transform=ax.get_yaxis_transform(), va="center")
ax.set_yticks([i[0] for i in items], [i[1] for i in items])
for lab, it in zip(ax.get_yticklabels(), items):
    lab.set_horizontalalignment("left")
    if it[2]:
        lab.set_fontweight("bold")
ax.tick_params(axis="y", length=0, pad=230)
ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
ax.set_xlim(0.5, 8)
ax.set_ylim(y - 0.7, 0.3)
ax.text(1.03, 0, "OR (95% CI)", transform=ax.get_yaxis_transform(), va="center", fontweight="bold")
ax.set_xlabel("Adjusted odds ratio for complication, open (OA) vs laparoscopic (LA), log scale")
ax.spines[["top", "right", "left"]].set_visible(False)
fig.text(0.01, 0.01, "OR > 1: more complications with OA. Adjusted for complicated appendicitis and comorbidity. "
         "Synthetic data for teaching; not real patients.", fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.04, 0.84, 1))
fig.savefig(fig_path, dpi=150)
print(f"\nsaved {csv_path}, {fig_path}")

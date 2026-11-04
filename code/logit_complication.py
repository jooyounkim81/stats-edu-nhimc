"""
logit_complication.py — 합병증(complication)에 대한 치료방법별 승산비: 보정 전후 로지스틱 회귀
사용법: python code/logit_complication.py [입력 CSV] [출력 CSV]
  기본 입력: data/appendicitis_synthetic.csv
  기본 출력: results/logit_complication.csv (UTF-8 BOM)
※ 강의용 합성데이터. 실제 환자 아님.

제외 기준: Table 1과 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸)
모형 1 (보정 전): complication ~ treatment
모형 2 (보정 후): complication ~ treatment + complicated + age_group + comorbidity + sex + year
  기준군: treatment = LA, age_group = 18-64, sex = F. year는 연속형(1년 증가당 승산비).
승산비 = exp(회귀계수), 95% 신뢰구간 = Wald 방식, p값 = Wald 검정
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = sys.argv[1] if len(sys.argv) > 1 else "data/appendicitis_synthetic.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/logit_complication.csv"

# 제외
df = pd.read_csv(src, encoding="utf-8-sig", dtype={"sex": str})
blank = df.sex.isna() | (df.sex.str.strip() == "")
d = df[(df.age <= 120) & (df.los_days >= 0) & ~blank].copy()
print(f"분석 대상 {len(d):,}명 (원자료 {len(df):,}명, 제외 {len(df) - len(d)}명), 합병증 {int(d.complication.sum())}건")

# 기준군 지정
d["treatment"] = pd.Categorical(d.treatment, ["LA", "OA", "NOM"])
d["age_group"] = pd.Categorical(d.age_group, ["18-64", "<18", ">=65"])
d["sex"] = pd.Categorical(d.sex.str.strip(), ["F", "M"])

models = {
    "보정 전": "complication ~ treatment",
    "보정 후": "complication ~ treatment + complicated + age_group + comorbidity + sex + year",
}
LABELS = {
    "treatment[T.OA]": "개복 수술(OA) vs 복강경(LA)",
    "treatment[T.NOM]": "비수술(NOM) vs 복강경(LA)",
    "complicated": "복잡성 충수염 (있음 vs 없음)",
    "age_group[T.<18]": "18세 미만 vs 18–64세",
    "age_group[T.>=65]": "65세 이상 vs 18–64세",
    "comorbidity": "동반질환 (있음 vs 없음)",
    "sex[T.M]": "남성 vs 여성",
    "year": "진단 연도 (1년 증가당)",
}

rows = []
for name, f in models.items():
    m = smf.logit(f, d).fit(disp=0)
    ci = m.conf_int()
    print(f"[{name}] n = {int(m.nobs):,}, 수렴 = {m.mle_retvals['converged']}")
    for term in m.params.index:
        if term == "Intercept":
            continue
        p = m.pvalues[term]
        rows.append({
            "모형": name, "변수": LABELS.get(term, term),
            "승산비": f"{np.exp(m.params[term]):.2f}",
            "95% CI 하한": f"{np.exp(ci.loc[term, 0]):.2f}",
            "95% CI 상한": f"{np.exp(ci.loc[term, 1]):.2f}",
            "p값": "<0.001" if p < 0.001 else f"{p:.3f}",
        })

table = pd.DataFrame(rows)
table.to_csv(dst, index=False, encoding="utf-8-sig")
print()
print(table.to_csv(index=False, sep="\t"), end="")
print(f"\nsaved {dst}")

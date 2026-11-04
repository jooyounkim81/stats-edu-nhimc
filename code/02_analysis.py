"""
02_analysis.py — 합성데이터 분석: 정제 → Table 1 → 결과변수 → 교란 보정 → 하위군 → 연도 추세
사용법: python 02_analysis.py [결과 JSON 파일명]
※ 강의용 합성데이터. 실제 환자 아님.
"""
import json
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from importlib import import_module

warnings.filterwarnings("ignore")
generate = import_module("01_generate").generate
SEEDS = [20261103] + list(range(20261104, 20261115))   # 강의용 시드 12개
TREATS = ["LA", "OA", "NOM"]


def r(x, k=2):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def orci(model, term):
    b = model.params[term]
    lo, hi = model.conf_int().loc[term]
    return {"or": r(np.exp(b)), "lo": r(np.exp(lo)), "hi": r(np.exp(hi)), "p": r(model.pvalues[term], 4)}


def analyze(seed):
    df = generate(seed)
    out = {"seed": seed}

    # ① 정제: 입력 오류 찾기
    bad_age, bad_los, miss_sex = int((df.age > 120).sum()), int((df.los_days < 0).sum()), int(df.sex.isna().sum())
    d = df[(df.age <= 120) & (df.los_days >= 0) & df.sex.notna()].copy()
    out["clean"] = {"n_raw": len(df), "bad_age": bad_age, "bad_los": bad_los, "miss_sex": miss_sex, "n": len(d)}
    d["old"] = (d.age_group == ">=65").astype(int)
    d["male"] = (d.sex == "M").astype(int)
    d["treatment"] = pd.Categorical(d.treatment, TREATS)
    d["age_group"] = pd.Categorical(d.age_group, ["18-64", "<18", ">=65"])

    # ② 치료법 연도별 비율
    ct = pd.crosstab(d.year, d.treatment, normalize="index") * 100
    out["trend"] = {int(y): {t: r(ct.loc[y, t], 1) for t in TREATS} for y in ct.index}

    # ③ Table 1 (치료법별)
    t1 = {}
    for t in TREATS + ["ALL"]:
        s = d if t == "ALL" else d[d.treatment == t]
        q = s.age.quantile([.25, .5, .75])
        t1[t] = {"n": len(s), "age_med": r(q[.5], 0), "age_q1": r(q[.25], 0), "age_q3": r(q[.75], 0),
                 "male": int(s.male.sum()), "lt18": int((s.age_group == "<18").sum()), "a1864": int((s.age_group == "18-64").sum()),
                 "ge65": int((s.age_group == ">=65").sum()), "comp": int(s.complicated.sum()), "cmb": int(s.comorbidity.sum())}
    out["table1"] = t1

    # ④ 결과변수 요약 (치료법별)
    o = {}
    for t in TREATS:
        s = d[d.treatment == t]
        o[t] = {"los_mean": r(s.los_days.mean(), 1), "los_med": r(s.los_days.median(), 1),
                "los_q1": r(s.los_days.quantile(.25), 1), "los_q3": r(s.los_days.quantile(.75), 1),
                "abx_med": r(s.antibiotic_days.median(), 1), "ana_med": r(s.analgesic_days.median(), 1),
                "ana_med_uncomp": r(s[s.complicated == 0].analgesic_days.median(), 1),
                "comp": r(s.complication.mean() * 100, 1), "readmit": r(s.readmit_30d.mean() * 100, 1), "fail": r(s.failure_1y.mean() * 100, 1)}
    out["outcomes"] = o
    hist = {}
    for t in TREATS:
        v = d[d.treatment == t].los_days.clip(upper=30)
        hist[t] = [int((v == k).sum()) for k in range(1, 31)]
    out["los_hist"] = hist
    out["los_all"] = {"mean": r(d.los_days.mean(), 1), "median": r(d.los_days.median(), 1), "skew": r(d.los_days.skew(), 2), "ge30": int((d.los_days >= 30).sum())}

    # ⑤ 합병증: 보정 전후 (교란)
    m0 = smf.logit("complication ~ treatment", d).fit(disp=0)
    m1 = smf.logit("complication ~ treatment + complicated + age_group + comorbidity + sex + year", d).fit(disp=0)
    out["complication"] = {"rate": r(d.complication.mean() * 100, 1),
                           "crude": {"OA": orci(m0, "treatment[T.OA]"), "NOM": orci(m0, "treatment[T.NOM]")},
                           "adj": {"OA": orci(m1, "treatment[T.OA]"), "NOM": orci(m1, "treatment[T.NOM]"),
                                   "complicated": orci(m1, "complicated"), "old": orci(m1, "age_group[T.>=65]"),
                                   "comorbidity": orci(m1, "comorbidity")}}
    # 재입원
    m = smf.logit("readmit_30d ~ treatment + complicated + age_group + comorbidity + sex", d).fit(disp=0)
    out["readmit"] = {"rate": r(d.readmit_30d.mean() * 100, 1), "complicated": orci(m, "complicated"), "NOM": orci(m, "treatment[T.NOM]")}
    # 1년 치료실패: 위험비와 승산비
    a = d[d.treatment == "NOM"].failure_1y; b = d[d.treatment == "LA"].failure_1y
    p1, p0 = a.mean(), b.mean()
    rr = p1 / p0
    se = np.sqrt((1 - p1) / a.sum() + (1 - p0) / b.sum())
    orv = (p1 / (1 - p1)) / (p0 / (1 - p0))
    se_or = np.sqrt(1 / a.sum() + 1 / (len(a) - a.sum()) + 1 / b.sum() + 1 / (len(b) - b.sum()))
    out["failure"] = {"NOM": r(p1 * 100, 1), "LA": r(p0 * 100, 1), "OA": r(d[d.treatment == "OA"].failure_1y.mean() * 100, 1),
                      "rr": r(rr, 1), "rr_lo": r(np.exp(np.log(rr) - 1.96 * se), 1), "rr_hi": r(np.exp(np.log(rr) + 1.96 * se), 1),
                      "or": r(orv, 1), "or_lo": r(np.exp(np.log(orv) - 1.96 * se_or), 1), "or_hi": r(np.exp(np.log(orv) + 1.96 * se_or), 1)}

    # ⑥ 하위군: 개복 vs 복강경 합병증 (연령·성별)
    sub = {}
    for key, mask in [("lt65", d.old == 0), ("ge65", d.old == 1), ("female", d.male == 0), ("male", d.male == 1)]:
        mm = smf.logit("complication ~ treatment + complicated + comorbidity", d[mask]).fit(disp=0)
        sub[key] = orci(mm, "treatment[T.OA]") | {"n": int(mask.sum())}
    sub["p_int_age"] = r(smf.logit("complication ~ treatment*old + complicated + comorbidity", d).fit(disp=0).pvalues["treatment[T.OA]:old"], 3)
    sub["p_int_sex"] = r(smf.logit("complication ~ treatment*male + complicated + comorbidity", d).fit(disp=0).pvalues["treatment[T.OA]:male"], 3)
    out["subgroup"] = sub
    # 성별 차이 검정 18번 (결과변수 6 × 연령군 3)
    tests = []
    for y in ["complication", "readmit_30d", "failure_1y", "los_days", "antibiotic_days", "analgesic_days"]:
        for g in ["<18", "18-64", ">=65"]:
            s = d[d.age_group == g]
            if y in ("los_days", "antibiotic_days", "analgesic_days"):
                p = smf.ols(f"np.log1p({y}) ~ male", s).fit().pvalues["male"]
            else:
                p = smf.logit(f"{y} ~ male", s).fit(disp=0).pvalues["male"]
            tests.append({"y": y, "g": g, "p": r(p, 3)})
    out["sex_tests"] = tests

    # ⑦ 연도 추세 (재원일수)
    out["year_med"] = {int(k): r(v, 1) for k, v in d.groupby("year").los_days.median().items()}
    c0 = smf.ols("np.log(los_days) ~ year", d).fit().params["year"]
    c1 = smf.ols("np.log(los_days) ~ year + treatment + complicated + age_group + comorbidity", d).fit().params["year"]
    out["year_trend"] = {"crude_pct": r((np.exp(c0) - 1) * 100, 1), "adj_pct": r((np.exp(c1) - 1) * 100, 1)}

    if seed == SEEDS[0]:
        prev = pd.concat([df.head(20), df[(df.age > 120) | (df.los_days < 0) | df.sex.isna()]])
        out["preview"] = json.loads(prev.to_json(orient="records"))
    return out


if __name__ == "__main__":
    res = [analyze(s) for s in SEEDS]
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    print(f"saved {path}: {len(res)} seeds")

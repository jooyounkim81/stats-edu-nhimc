"""
01_generate.py — 합성데이터 생성: 급성 충수염 2016~2025, 연 200명, 총 2,000명
※ 강의용 가상 자료. 실제 환자 아님. 임상 근거 아님.
사용법: python 01_generate.py [시드] [저장할 파일명]
"""
import sys
import numpy as np
import pandas as pd

YEARS = list(range(2016, 2026))
N_PER_YEAR = 200


def expit(x):
    return 1 / (1 + np.exp(-x))


def generate(seed=20261103):
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for y in YEARS:
        for _ in range(N_PER_YEAR):
            pid += 1
            g = rng.choice(["<18", "18-64", ">=65"], p=[0.25, 0.55, 0.20])
            age = int(rng.integers(5, 18)) if g == "<18" else int(rng.integers(18, 65)) if g == "18-64" else int(rng.integers(65, 91))
            sex = "M" if rng.random() < 0.52 else "F"
            p_comp = {"<18": 0.25, "18-64": 0.15, ">=65": 0.35}[g]
            p_cmb = {"<18": 0.03, "18-64": 0.15, ">=65": 0.50}[g]
            rows.append(dict(patient_id=f"P{pid:04d}", year=y, age=age, age_group=g, sex=sex,
                             complicated=int(rng.random() < p_comp), comorbidity=int(rng.random() < p_cmb)))
    df = pd.DataFrame(rows)
    old = (df.age_group == ">=65").astype(int)

    # 치료법 선택: 중증·고령·동반질환이면 개복이 늘도록. 연도별 목표 비율(복강경 55→85%, 개복 35→5%, 비수술 10%)
    lin_oa = 1.4 * df.complicated + 0.7 * old + 0.6 * df.comorbidity
    lin_nom = -3.0 * df.complicated
    target_oa = dict(zip(YEARS, np.linspace(0.35, 0.05, len(YEARS))))
    treat = np.empty(len(df), dtype=object)
    for y in YEARS:
        m = (df.year == y).values
        a = b = 0.0
        for _ in range(500):
            eo, en = np.exp(a + lin_oa[m]), np.exp(b + lin_nom[m])
            den = 1 + eo + en
            a += np.log(target_oa[y] / (eo / den).mean()) * 0.5
            b += np.log(0.10 / (en / den).mean()) * 0.5
        eo, en = np.exp(a + lin_oa[m]), np.exp(b + lin_nom[m]); den = 1 + eo + en
        probs = np.column_stack([1 / den, eo / den, en / den])
        treat[m] = [rng.choice(["LA", "OA", "NOM"], p=p) for p in probs]
    df["treatment"] = treat
    OA, NOM = (df.treatment == "OA").astype(int), (df.treatment == "NOM").astype(int)

    # T1 재원일수 (오른쪽으로 치우침, 1% 장기 입원)
    mu = 3 + 2 * OA + 1 * NOM + 3 * df.complicated + 1.5 * old + 1 * df.comorbidity - 0.1 * (df.year - 2016)
    los = np.maximum(1, np.round(mu * np.exp(rng.normal(0, 0.35, len(df)))))
    los = los + (rng.random(len(df)) < 0.01) * rng.integers(20, 41, len(df))
    df["los_days"] = los.astype(int)
    # T2 항생제 투여일수
    df["antibiotic_days"] = np.where(NOM == 1, rng.integers(7, 11, len(df)),
                                     np.where(df.complicated == 1, rng.integers(5, 8, len(df)), rng.integers(1, 3, len(df))))
    # T3 진통제 사용일수
    df["analgesic_days"] = np.maximum(0, np.round(2 + 1.5 * OA + 1 * df.complicated + rng.normal(0, 1, len(df)))).astype(int)
    # T4·T7 합병증
    lp = (np.log(0.05 / 0.95) + np.log(1.5) * OA + np.log(2.5 / 1.5) * OA * old
          + np.log(3.5) * df.complicated + np.log(1.8) * old + np.log(0.8) * NOM)
    df["complication"] = (rng.random(len(df)) < expit(lp)).astype(int)
    # T5 30일 재입원
    lp = np.log(0.04 / 0.96) + np.log(2) * df.complicated + np.log(1.5) * NOM
    df["readmit_30d"] = (rng.random(len(df)) < expit(lp)).astype(int)
    # T6 1년 치료실패
    df["failure_1y"] = (rng.random(len(df)) < np.where(NOM == 1, 0.25, 0.02)).astype(int)
    # T8 성별: 어떤 결과에도 효과를 넣지 않음
    # T10 입력 오류 7건
    err = rng.choice(len(df), 7, replace=False)
    df.loc[err[:2], "age"] = 150
    df.loc[err[2:4], "los_days"] = -3
    df["sex"] = df["sex"].astype(object)
    df.loc[err[4:7], "sex"] = np.nan
    return df


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261103
    out = sys.argv[2] if len(sys.argv) > 2 else "appendicitis_synthetic.csv"
    generate(seed).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {out}, seed {seed}")

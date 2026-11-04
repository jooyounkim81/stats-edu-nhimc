"""
table1.py — 치료방법(treatment)별 대상자 특성표(Table 1)
사용법: python code/table1.py [입력 CSV] [출력 TSV(선택)]
  기본 입력: data/appendicitis_synthetic.csv
  출력 TSV를 지정하면 탭으로 구분한 표를 저장한다(엑셀에 그대로 붙여 넣을 수 있음).
※ 강의용 합성데이터. 실제 환자 아님.

① 입력 오류 점검: 나이(0 미만 또는 120 초과, 연령군 불일치), 재원일수(음수), 모든 열의 빈칸
② 제외 기준: 02_analysis.py와 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸)
③ 연속형: 중앙값 (사분위범위 Q1–Q3), 범주형: n (%), %의 분모는 각 치료군 인원
"""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/appendicitis_synthetic.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else None
TREATS = ["LA", "OA", "NOM"]

# ① 입력 오류 점검 (빈칸은 문자열로 읽어 직접 센다)
raw = pd.read_csv(src, encoding="utf-8-sig", dtype=str, keep_default_na=False)
df = pd.read_csv(src, encoding="utf-8-sig")
print(f"[점검] 원자료 {len(df):,}명, 변수 {df.shape[1]}개")

blank = {c: int((raw[c].str.strip() == "").sum()) for c in raw.columns}
print("[점검] 빈칸:", {c: n for c, n in blank.items() if n} or "없음")

bad_age = df[(df.age < 0) | (df.age > 120)]
print(f"[점검] 나이 범위 밖(0 미만 또는 120 초과): {len(bad_age)}건", bad_age[["patient_id", "age"]].values.tolist())
ag = pd.cut(df.age, [-1, 17, 64, 10**6], labels=["<18", "18-64", ">=65"]).astype(str)
mism = df[ag != df.age_group]
print(f"[점검] 나이와 연령군(age_group) 불일치: {len(mism)}건", mism[["patient_id", "age", "age_group"]].values.tolist())

bad_los = df[df.los_days < 0]
print(f"[점검] 재원일수 음수: {len(bad_los)}건", bad_los[["patient_id", "los_days"]].values.tolist())
print(f"[점검] 재원일수 범위: {df.los_days.min()}–{df.los_days.max()}일, 30일 이상 {int((df.los_days >= 30).sum())}건(값 자체는 가능하므로 제외하지 않음)")
print(f"[점검] 환자번호 중복: {int(df.patient_id.duplicated().sum())}건")

# ② 제외
keep = (df.age <= 120) & (df.los_days >= 0) & df.sex.notna()
d = df[keep].copy()
print(f"[정제] 제외 {int((~keep).sum())}명 → 분석 대상 {len(d):,}명\n")

# ③ Table 1
groups = [("전체", d)] + [(t, d[d.treatment == t]) for t in TREATS]


def med_iqr(s):
    q = s.quantile([.25, .5, .75])
    return f"{q[.5]:.0f} ({q[.25]:.0f}–{q[.75]:.0f})"


def n_pct(mask, s):
    n = int(mask.sum())
    return f"{n} ({n / len(s) * 100:.1f})"


rows = [["변수"] + [f"{name} (n={len(s):,})" for name, s in groups]]
rows.append(["나이, 세, 중앙값 (IQR)"] + [med_iqr(s.age) for _, s in groups])
rows.append(["연령군, n (%)"] + [""] * len(groups))
for lab, g in [("  18세 미만", "<18"), ("  18–64세", "18-64"), ("  65세 이상", ">=65")]:
    rows.append([lab] + [n_pct(s.age_group == g, s) for _, s in groups])
rows.append(["남성, n (%)"] + [n_pct(s.sex == "M", s) for _, s in groups])
rows.append(["복잡성 충수염, n (%)"] + [n_pct(s.complicated == 1, s) for _, s in groups])
rows.append(["동반질환, n (%)"] + [n_pct(s.comorbidity == 1, s) for _, s in groups])

table = "\n".join("\t".join(r) for r in rows)
print(table)
print("\nLA: 복강경 수술, OA: 개복 수술, NOM: 비수술 항생제 치료. IQR: 사분위범위(Q1–Q3).")
if dst:
    with open(dst, "w", encoding="utf-8-sig") as f:
        f.write(table + "\n")
    print(f"saved {dst}")

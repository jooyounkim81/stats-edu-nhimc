"""
table2.py — 치료방법(treatment)별 결과 요약표(Table 2)
사용법: python code/table2.py [입력 CSV] [출력 CSV]
  기본 입력: data/appendicitis_synthetic.csv
  기본 출력: results/table2_outcomes.csv (UTF-8 BOM, 엑셀에서 한글이 깨지지 않음)
※ 강의용 합성데이터. 실제 환자 아님.

① 제외 기준: Table 1과 같음 (나이 > 120, 재원일수 < 0, 성별 빈칸)
② 일수 변수: 평균, 중앙값 (사분위범위 Q1–Q3), 소수 첫째 자리
③ 예/아니오 변수: 발생 수 (%), %의 분모는 각 치료군 인원
"""
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/appendicitis_synthetic.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "results/table2_outcomes.csv"
TREATS = ["LA", "OA", "NOM"]
DAYS = [("los_days", "재원일수, 일"), ("antibiotic_days", "항생제 투여일수, 일"), ("analgesic_days", "진통제 사용일수, 일")]
EVENTS = [("complication", "합병증"), ("readmit_30d", "30일 내 재입원"), ("failure_1y", "1년 내 치료실패")]

# ① 제외
df = pd.read_csv(src, encoding="utf-8-sig", dtype={"sex": str})
sex_blank = df.sex.isna() | (df.sex.str.strip() == "")
flags = {"나이 120세 초과": df.age > 120, "재원일수 0 미만": df.los_days < 0, "성별 빈칸": sex_blank}
for k, v in flags.items():
    print(f"[제외] {k}: {int(v.sum())}건")
drop = flags["나이 120세 초과"] | flags["재원일수 0 미만"] | flags["성별 빈칸"]
d = df[~drop].copy()
print(f"[제외] 합계 {int(drop.sum())}건 → 분석 대상 {len(d):,}명 (원자료 {len(df):,}명)")

# 결과 변수 빈칸 확인 (있으면 알림, 해당 변수에서만 빠짐)
for c, _ in DAYS + EVENTS:
    if d[c].isna().any():
        print(f"※ {c}: 빈칸 {int(d[c].isna().sum())}건 (해당 변수 요약에서 제외)")

groups = [("전체", d)] + [(t, d[d.treatment == t]) for t in TREATS]

# ②·③ 표 작성
rows = []
rows.append({"변수": "대상자 수", "통계량": "n", **{f"{g}": f"{len(s):,}" for g, s in groups}})
for c, lab in DAYS:
    rows.append({"변수": lab, "통계량": "평균", **{g: f"{s[c].mean():.1f}" for g, s in groups}})
    rows.append({"변수": lab, "통계량": "중앙값 (Q1–Q3)",
                 **{g: f"{s[c].median():.1f} ({s[c].quantile(.25):.1f}–{s[c].quantile(.75):.1f})" for g, s in groups}})
for c, lab in EVENTS:
    rows.append({"변수": lab, "통계량": "n (%)",
                 **{g: f"{int(s[c].sum())} ({s[c].mean() * 100:.1f})" for g, s in groups}})

table = pd.DataFrame(rows)
table.to_csv(dst, index=False, encoding="utf-8-sig")
print()
print(table.to_csv(index=False, sep="\t"), end="")
print("\nLA: 복강경 수술, OA: 개복 수술, NOM: 비수술 항생제 치료. Q1–Q3: 사분위범위.")
print(f"saved {dst}")

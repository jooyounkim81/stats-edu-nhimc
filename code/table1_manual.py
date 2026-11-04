# -*- coding: utf-8 -*-
# 급성 충수염 합성데이터 — 치료방법(treatment)별 대상자 특성표 작성
# 의존성: pandas만 사용 (p값 등 통계 검정은 scipy가 필요하므로 포함하지 않음)

import pandas as pd  # 자료 처리를 위해 pandas를 불러옴

# =========================================================
# 0. 설정값
# =========================================================
DATA_PATH = "data/appendicitis_synthetic.csv"  # 원자료 CSV 경로
OUT_PATH = "data/table1_by_treatment.csv"  # 특성표를 저장할 CSV 경로
GROUP_VAR = "treatment"  # 집단 구분 변수(치료방법)

BASELINE_CONT = ["age"]  # 기저 특성 — 연속형 변수
BASELINE_CAT = ["year", "age_group", "sex", "complicated", "comorbidity"]  # 기저 특성 — 범주형 변수
OUTCOME_CONT = ["los_days", "antibiotic_days", "analgesic_days"]  # 치료 경과 — 연속형 변수
OUTCOME_CAT = ["complication", "readmit_30d", "failure_1y"]  # 치료 결과 — 범주형 변수

pd.set_option("display.max_rows", None)  # 출력 시 행을 생략하지 않도록 설정
pd.set_option("display.max_columns", None)  # 출력 시 열을 생략하지 않도록 설정
pd.set_option("display.width", 200)  # 콘솔 출력 너비를 넓게 설정

# =========================================================
# 1. 자료 불러오기
# =========================================================
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype={"sex": str})  # CSV를 읽고 성별은 문자열로 읽음(한글이 깨지면 encoding="cp949"로 변경)
n_raw = len(df)  # 원자료 전체 행 수 저장
print(f"[원자료] 전체 {n_raw:,}건")  # 원자료 건수 출력

for col in BASELINE_CONT + OUTCOME_CONT:  # 연속형 변수마다 반복
    before_na = df[col].isna().sum()  # 숫자 변환 전 결측 수
    df[col] = pd.to_numeric(df[col], errors="coerce")  # 숫자로 변환(변환 불가 값은 결측 처리)
    added_na = df[col].isna().sum() - before_na  # 변환 과정에서 새로 생긴 결측 수
    if added_na > 0:  # 숫자가 아닌 값이 있었다면
        print(f"  ※ {col}: 숫자로 변환할 수 없는 값 {added_na}건 → 결측 처리")  # 경고 출력

# =========================================================
# 2. 이상값·결측 점검 및 제외
# =========================================================
flag_age = df["age"] > 120  # 나이 120세 초과 여부(결측은 False)
flag_los = df["los_days"] < 0  # 재원일수 0 미만 여부(결측은 False)
flag_sex = df["sex"].isna() | (df["sex"].str.strip() == "")  # 성별 빈칸 여부(결측 또는 공백 문자만 있는 경우)
flag_any = flag_age | flag_los | flag_sex  # 세 조건 중 하나라도 해당하는지 여부
n_multi = int(((flag_age.astype(int) + flag_los.astype(int) + flag_sex.astype(int)) >= 2).sum())  # 두 조건 이상에 중복 해당하는 건수

print("\n[제외 기준 점검]")  # 점검 결과 제목 출력
print(f"  나이 120세 초과      : {flag_age.sum():,}건")  # 나이 이상값 건수
print(f"  재원일수 0 미만      : {flag_los.sum():,}건")  # 재원일수 이상값 건수
print(f"  성별 빈칸            : {flag_sex.sum():,}건")  # 성별 빈칸 건수
print(f"  (두 조건 이상 중복   : {n_multi:,}건)")  # 중복 해당 건수
print(f"  제외 합계(중복 제거) : {flag_any.sum():,}건")  # 실제로 제외되는 행 수

df = df.loc[~flag_any].copy()  # 해당 행을 제외한 분석용 자료 생성
df["sex"] = df["sex"].str.strip()  # 성별 값의 앞뒤 공백 제거
print(f"  분석 대상            : {len(df):,}건 (원자료 {n_raw:,}건 중)")  # 남은 분석 대상 건수

# =========================================================
# 3. 집단 구성
# =========================================================
n_trt_na = int(df[GROUP_VAR].isna().sum())  # 치료방법 결측 건수
if n_trt_na > 0:  # 치료방법이 비어 있는 행이 있으면
    print(f"  ※ treatment 결측 {n_trt_na}건은 '전체' 열에만 포함됨")  # 안내 출력

groups = sorted(df[GROUP_VAR].dropna().unique().tolist(), key=str)  # 치료방법 범주 목록(정렬)
subsets = {str(g): df[df[GROUP_VAR] == g] for g in groups}  # 치료방법별 하위 자료 사전
subsets["전체"] = df  # 마지막 열로 전체 자료 추가


# =========================================================
# 4. 요약 함수
# =========================================================
def mean_sd(s):  # 평균 ± 표준편차 문자열을 만드는 함수
    s = s.dropna()  # 결측 제외
    if len(s) == 0:  # 유효값이 없으면
        return "-"  # 대시 반환
    sd = s.std()  # 표본 표준편차 계산(n-1 기준)
    return f"{s.mean():.1f} ± {sd:.1f}" if pd.notna(sd) else f"{s.mean():.1f}"  # n=1이라 SD 계산이 불가하면 평균만 표시


def median_iqr(s):  # 중앙값 [Q1, Q3] 문자열을 만드는 함수
    s = s.dropna()  # 결측 제외
    if len(s) == 0:  # 유효값이 없으면
        return "-"  # 대시 반환
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]"  # 중앙값과 제1·제3사분위수 표시


def add_cont(rows, section, var):  # 연속형 변수의 행을 표에 추가하는 함수
    for label, func in [("평균 ± SD", mean_sd), ("중앙값 [Q1, Q3]", median_iqr)]:  # 두 가지 요약 방식을 차례로 적용
        row = {"구분": section, "변수": var, "범주": "", "통계량": label}  # 행의 기본 정보
        for name, sub in subsets.items():  # 집단마다 반복
            row[name] = func(sub[var])  # 해당 집단의 요약값 계산
        rows.append(row)  # 행 목록에 추가
    if df[var].isna().any():  # 결측이 하나라도 있으면
        row = {"구분": section, "변수": var, "범주": "결측", "통계량": "n"}  # 결측 행의 기본 정보
        for name, sub in subsets.items():  # 집단마다 반복
            row[name] = str(int(sub[var].isna().sum()))  # 집단별 결측 수
        rows.append(row)  # 행 목록에 추가


def add_cat(rows, section, var):  # 범주형 변수의 행을 표에 추가하는 함수
    levels = sorted(df[var].dropna().unique().tolist(), key=str)  # 범주 목록(정렬)
    for lv in levels:  # 범주마다 반복
        row = {"구분": section, "변수": var, "범주": str(lv), "통계량": "n (%)"}  # 행의 기본 정보
        for name, sub in subsets.items():  # 집단마다 반복
            n = int((sub[var] == lv).sum())  # 해당 범주의 빈도
            pct = n / len(sub) * 100 if len(sub) > 0 else 0.0  # 집단 전체 인원 대비 백분율
            row[name] = f"{n} ({pct:.1f})"  # "빈도 (백분율)" 형식으로 기록
        rows.append(row)  # 행 목록에 추가
    if df[var].isna().any():  # 결측이 하나라도 있으면
        row = {"구분": section, "변수": var, "범주": "결측", "통계량": "n (%)"}  # 결측 행의 기본 정보
        for name, sub in subsets.items():  # 집단마다 반복
            n = int(sub[var].isna().sum())  # 결측 빈도
            pct = n / len(sub) * 100 if len(sub) > 0 else 0.0  # 집단 전체 인원 대비 백분율
            row[name] = f"{n} ({pct:.1f})"  # "빈도 (백분율)" 형식으로 기록
        rows.append(row)  # 행 목록에 추가


# =========================================================
# 5. 특성표 작성
# =========================================================
rows = []  # 표의 행을 담을 빈 목록
n_row = {"구분": "", "변수": "대상자 수", "범주": "", "통계량": "N"}  # 집단별 대상자 수 행
for name, sub in subsets.items():  # 집단마다 반복
    n_row[name] = str(len(sub))  # 집단 크기 기록
rows.append(n_row)  # 첫 행으로 추가

for v in BASELINE_CONT:  # 기저 특성 연속형 변수마다
    add_cont(rows, "기저 특성", v)  # 연속형 행 추가
for v in BASELINE_CAT:  # 기저 특성 범주형 변수마다
    add_cat(rows, "기저 특성", v)  # 범주형 행 추가
for v in OUTCOME_CONT:  # 치료 경과 연속형 변수마다
    add_cont(rows, "치료 경과·결과", v)  # 연속형 행 추가
for v in OUTCOME_CAT:  # 치료 결과 범주형 변수마다
    add_cat(rows, "치료 경과·결과", v)  # 범주형 행 추가

table1 = pd.DataFrame(rows)  # 행 목록을 데이터프레임으로 변환

# =========================================================
# 6. 출력 및 저장
# =========================================================
print("\n[치료방법별 대상자 특성표]")  # 표 제목 출력
print(table1.to_string(index=False))  # 표 전체를 인덱스 없이 출력
table1.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")  # 엑셀에서 한글이 깨지지 않도록 BOM 포함 UTF-8로 저장
print(f"\n저장 완료: {OUT_PATH}")  # 저장 위치 안내

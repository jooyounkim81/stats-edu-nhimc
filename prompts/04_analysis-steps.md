# 단계별 분석 지시문 (Claude Code 등 코드를 실행하는 AI용)

강의에서 Claude Code 웹에 차례로 넣은 지시문입니다. 같은 세션에서 순서대로 넣으세요.
결과가 나오면 숫자와 설명을 그대로 믿지 말고, 저장소 맨 위 README의 검증 질문으로 확인하세요.

## 0. 대상자 특성 (Table 1)

```
data/appendicitis_synthetic.csv 를 읽어서 치료방법(treatment)별 대상자 특성표(Table 1)를 만들어 줘.
- 연속형 변수는 중앙값(사분위범위), 범주형 변수는 n(%)로
- 분석 전에 이상한 값(나이, 재원일수, 빈칸)이 있는지 먼저 찾아서 몇 건인지 알려 줘
- 실행한 코드는 code/table1.py 로 저장해 줘
```

## 1. 결과변수 요약 (Table 2)

```
data/appendicitis_synthetic.csv 로 치료방법(treatment)별 결과 요약표를 만들어 줘.
- 먼저 나이 120세 초과, 재원일수 0 미만, 성별 빈칸인 행을 제외해 줘 (Table 1과 같은 기준)
- 일수 변수(los_days, antibiotic_days, analgesic_days): 평균, 중앙값, 사분위범위
- 예/아니오 변수(complication, readmit_30d, failure_1y): 발생 수와 비율(%)
- 결과는 results/table2_outcomes.csv, 코드는 code/table2.py 로 저장해 줘
```

## 2. 일수 결과의 분석 방법 고르기

```
재원일수(los_days)를 치료방법별로 비교하려고 해.
분석을 바로 실행하지 말고, 먼저 아래를 알려 줘.
1. 재원일수 분포가 어떤 모양인지 (히스토그램을 results/figures/los_hist.png 로 저장, 그림 글자는 영어로)
2. 이 분포에 맞는 비교 방법 2가지와 각각의 가정
3. 평균을 비교하는 방법(t검정, 분산분석)을 그대로 쓰면 문제가 되는지, 된다면 왜인지
```

## 3. 합병증, 보정 전후

```
합병증(complication)에 대해 로지스틱 회귀로 개복(OA)과 비수술(NOM)의 승산비를 구해 줘. 기준군은 복강경(LA).
1. 보정 전: 치료방법만 넣은 모형
2. 보정 후: 치료방법 + complicated + age_group(기준 18-64) + comorbidity + sex + year(연속형)
승산비, 95% 신뢰구간, p값을 표로 보여 주고, 보정 전후에 승산비가 왜 달라졌는지 통계를 모르는 사람도 알 수 있게 설명해 줘.
코드는 code/logit_complication.py, 결과는 results/logit_complication.csv 로 저장해 줘.
```

## 4. 연령·성별 하위군

```
개복(OA) 대 복강경(LA)의 합병증 승산비가 연령과 성별에 따라 다른지 보려고 해.
1. 65세 미만과 65세 이상으로 나눠 각각 로지스틱 회귀 (보정: complicated, comorbidity)
2. 여성과 남성으로 나눠 같은 방법으로
3. 상호작용 검정: complication ~ treatment × (65세 이상 여부) + complicated + comorbidity, 성별도 같은 방식으로
4. 하위군 결과를 포리스트 그림으로 results/figures/subgroup_forest.png 에 저장 (그림 글자는 영어로)
결과를 해석해 줘. 코드는 code/subgroup.py 로 저장해 줘.
```

## 5. 치료실패, 위험비와 승산비

```
1년 치료실패(failure_1y)를 비수술(NOM)과 복강경(LA)으로 비교해서 위험비(RR)와 승산비(OR)를 각각 95% 신뢰구간과 함께 구해 줘.
두 값이 왜 다른지, 그리고 두 군에서 '치료실패'가 같은 뜻인지도 설명해 줘.
```

## 6. 연도 추세

```
재원일수가 연도에 따라 줄었는지 보려고 해.
1. 연도별 재원일수 중앙값 그래프를 results/figures/los_trend.png 로 저장 (그림 글자는 영어로)
2. log(재원일수) ~ year 회귀로 연간 변화율(%)
3. 치료방법, complicated, age_group, comorbidity를 보정한 뒤의 연간 변화율(%)
보정 전후 차이가 무엇을 뜻하는지 설명해 줘. 코드는 code/trend.py 로 저장해 줘.
```

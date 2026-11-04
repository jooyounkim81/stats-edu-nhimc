# 통계를 모르는 내가 어떻게 논문을?

AI를 활용한 통계분석 및 연구방법 — 직원 대상 통계교육 실습 자료

> **합성데이터입니다.** 실제 환자 자료가 아니며, 결과는 임상 근거가 아닙니다.
> 실제 환자 자료는 이 저장소나 어떤 AI에도 올리지 않습니다.

## 내용

| 경로 | 설명 |
|---|---|
| data/appendicitis_synthetic.csv | 급성 충수염 가상 환자 2,000명 (2016–2025, 연 200명, 시드 20261103) |
| code/01_generate.py | 합성데이터 생성 (시드를 고정하면 언제나 같은 자료) |
| code/02_analysis.py | 정제, 대상자 특성, 합병증 보정 전후, 하위군, 연도 추세 분석 (시드 12개) |
| results/results.json | 02_analysis.py 실행 결과 |
| prompts/ | 강의에서 AI에 넣은 지시문 |

## 실행

저장소 맨 위 폴더에서:

```
pip install -r requirements.txt
python code/01_generate.py 20261103 data/appendicitis_synthetic.csv
python code/02_analysis.py results/results.json
```

확인 값 (시드 20261103): 분석 대상 1,993명, 합병증에 대한 개복 수술 보정 후 승산비 2.34 (95% CI 1.61–3.41)

실행 확인 환경: Python 3.13.18, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0

## 변수

| 변수 | 뜻 |
|---|---|
| treatment | LA 복강경 수술, OA 개복 수술, NOM 비수술 항생제 치료 |
| complicated | 복잡성 충수염(천공·농양) 1/0 |
| comorbidity | 동반질환 1/0 |
| los_days | 재원일수 |
| antibiotic_days | 항생제 투여일수 |
| analgesic_days | 진통제 사용일수 |
| complication | 합병증 1/0 |
| readmit_30d | 30일 내 재입원 1/0 |
| failure_1y | 1년 내 치료실패 1/0 |

자료에는 연습용 입력 오류 7건(나이 150세, 재원일수 음수, 성별 빈칸)이 일부러 들어 있습니다.

정답표(심어 둔 효과)는 강의 후 공개합니다.

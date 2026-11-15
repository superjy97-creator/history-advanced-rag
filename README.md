# 한국사 Advanced RAG 사료 해설 시스템

고대부터 현대까지의 한국사 질문에 대해 실제 역사 사료를 검색하고, 근거가 확인된 내용만
한국어로 해설하는 수업 과제 프로젝트입니다. 역사학 전공자와 일반 학습자가 사료 원문,
출처, 이용조건을 함께 확인할 수 있도록 설계했습니다.

## 문서 안내

- [AI 답변 및 검증 지침](docs/AI_답변_및_검증_지침.md): 실제 프롬프트와 Python 검증 조건
- [Advanced RAG 과제 지침](docs/Advanced_RAG_과제_지침.md): 확인된 수업 요구사항과 구현 범위
- [Codex 유지보수 지침](AGENTS.md): 이후 안전한 수정과 검증을 위한 저장소 규칙

## 핵심 구조

질문은 다음 순서로 처리됩니다.

1. **Query Transformation — Gemini**: 인물·시대·사건을 분석하고 검색 질의를 보완합니다.
2. **Retrieval 및 Reranking — Gemini**: Pinecone과 공식 사료 검색 결과에서 질문의 근거가
   되는 원문만 선택합니다.
3. **Grounded Generation — GPT**: 선택된 사료만 사용해 한국어 해설과 직접 인용을 만듭니다.
4. **Groundedness Verification — GPT**: 주장과 인용을 원문에 대조하고, 근거가 부족하면
   답변을 보류하거나 검토 필요로 표시합니다.

모델 역할은 다음과 같이 고정되어 있습니다.

- 질의 변환·리랭킹: `gemini-3.5-flash-lite`
- 답변 생성·근거 검증: `gpt-6-luna`
- 문서·질문 임베딩: `text-embedding-3-large` (3,072차원)

## 하이브리드 검색과 Function Calling

Gemini가 필요에 따라 다음 Python 검색 도구를 요청하고, 애플리케이션이 실제로 실행한 뒤
결과를 모델에 돌려줍니다.

- `search_pinecone_history`: Pinecone의 14종 한국사 사료 통합 검색
- `search_official_history`: 국사편찬위원회 조선왕조실록의 공식 검색 페이지에서 국역·원문과
  고정 기사 URL 확인

공식 사이트 검색이 실패해도 Pinecone 검색은 독립적으로 계속 작동합니다. 실시간 검색 결과는
Pinecone에 다시 저장하지 않습니다.

## 데이터와 검증 현황

- 사료: 14종, 원문 기사 67,357개
- 임베딩 청크: 68,150개
- Pinecone: `history-advanced-rag` / `korean-history-v1`
- 인덱스: Dense, 3,072차원, cosine, AWS `us-east-1`
- 원격 ID 68,150개와 14종별 건수 일치 확인
- 자료별 대표 레코드 42건의 ID·본문·제목·출처 메타데이터 확인
- Ruff 통과, Pyright 오류 0건, 단위 테스트 47개 통과
- 고대·고려·조선·근대·현대 질문에서 실제 검색과 생성·검증 실행

사용한 사료는 삼국사기, 삼국유사, 한국고대금석문, 고려사, 조선왕조실록, 조선시대
법령자료, 고종시대사, 한국근대사자료집성, 3·1운동 격문, 국외항일운동자료,
자료대한민국사, 일제침략하 한국36년사, 한민족독립운동사자료집,
반민특위조사기록입니다.

대용량 원본과 전처리 JSONL은 GitHub에 포함하지 않습니다. 실행 앱은 이미 검증·적재된
Pinecone 데이터와 공식 사료 검색을 사용합니다.

## 현재 한계

- 공식 실시간 검색은 현재 조선왕조실록 한 곳에 연결되어 있습니다.
- 일부 자료의 링크는 개별 기사 URL이 아니라 공공데이터포털의 데이터셋 소개 페이지입니다.
  화면에서는 두 링크 유형을 구분해 표시합니다.
- 14종 사료가 모든 한국사 사건을 균등하게 포함하는 것은 아닙니다. 검색 근거가 부족하면
  답변을 생성하지 않습니다.
- 외부 모델·Pinecone·공식 사이트의 장애, 사용량 한도와 요금에 영향을 받습니다.

## 로컬 설치 및 실행

Python 3.11을 사용합니다. 프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

다음 환경변수가 필요합니다. 실제 키를 저장한 `.env`와 Streamlit Secrets는 Git에 올리지
않습니다.

```text
OPENAI_API_KEY
GEMINI_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME=history-advanced-rag
PINECONE_NAMESPACE=korean-history-v1
OPENAI_CHAT_MODEL=gpt-6-luna
GEMINI_MODEL=gemini-3.5-flash-lite
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

현재 로컬 구성은 프로젝트 상위 폴더의 `.env`도 읽으며, 모든 값은 최종적으로 운영체제
환경변수에서 가져옵니다.

## 검사 명령

```powershell
python -m unittest discover -s tests -v
python -m ruff check src scripts tests app.py
python -m pyright
```

Ruff와 Pyright는 개발 검사용 도구이므로 Streamlit 배포용 `requirements.txt`에는 포함하지
않았습니다.

## Streamlit Cloud 배포 구성

1. GitHub 저장소의 메인 파일로 `app.py`를 선택합니다.
2. Python 3.11을 선택합니다.
3. Streamlit Cloud의 Secrets 화면에 위 환경변수를 입력합니다.
4. API 키, `.env`, `.streamlit/secrets.toml`, 원본 데이터는 저장소에 올리지 않습니다.
5. 배포 후 실제 질문으로 Pinecone 검색, 공식 사료 검색, 출처 표시와 검증 결과를 확인합니다.

`requirements.txt`의 `-e .` 항목이 현재 저장소를 패키지로 설치하므로 `src/history_rag`를
Streamlit Cloud에서도 import할 수 있습니다.

## 데이터 출처와 이용조건

주요 제공기관은 교육부 국사편찬위원회이며, 공공데이터포털의 개별 데이터셋 상세 페이지에서
공공누리 제1유형과 인공지능 학습용 표시가 확인된 자료만 적재 대상으로 사용했습니다.

이용 시 다음 조건을 지킵니다.

- 제공기관·자료명·출처 URL·라이선스 표시
- 사료 직접 인용은 검색된 원문과 문자열 대조
- 개별 원문 URL과 데이터셋 소개 URL을 화면에서 구분
- 이용조건이 확인되지 않은 자료는 외부 API나 Pinecone으로 전송하지 않음
- 원문·번역·사진 등 제3자 권리가 별도로 표시된 경우 해당 조건을 우선 확인

외부 OpenAI API 전송과 Pinecone 저장 가능 여부는 특정 업체명이 명시된 개별 허가가 아니라
공공누리 제1유형 및 AI 활용 표시를 근거로 한 조건부 판단입니다. 자세한 조사 근거는 공개가
승인된 권한 요약 보고서와 최종 권한 대장에서 확인할 수 있습니다.


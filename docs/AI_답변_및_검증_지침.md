# AI 답변 및 검증 지침

이 문서는 현재 구현된 AI 동작을 코드 기준으로 설명한다. 실행 시 모델이 자동으로 읽는
프롬프트 파일이 아니며, 실제 규칙의 기준은 아래에 적은 Python 소스다.

## 처리 순서와 모델 역할

`AdvancedRAGPipeline.run()`은 **질의 변환 → 검색 → 리랭킹 → 생성 → 검증**을 순서대로
실행한다 (`src/history_rag/pipeline.py`). `RoutedHistoryLLM`은 질의 변환·리랭킹을 Gemini에,
생성·의미 검증을 GPT에 배정한다 (`src/history_rag/llm.py`).

1. 질의 변환: 질문의 고유명사·문헌명·날짜·번호를 보존하고, 검색 질의를 최대 3개로 만든다.
2. 검색: Gemini가 요청한 Pinecone/공식 사료 도구를 Python이 실행한다.
3. 리랭킹: 질문의 답 또는 일부가 원문에 명시된 후보만 남긴다.
4. 생성: 남은 사료 밖의 지식·수치·추측을 사용하지 않고 답과 직접 인용을 만든다.
5. 검증: 인용 문자열과 출처 ID를 코드로 확인하고, 별도 GPT 호출로 주장 단위 의미 검증을 한다.

## 모델 프롬프트의 핵심 규칙

실제 프롬프트는 `OpenAIHistoryLLM`의 각 메서드 안에 있다
(`src/history_rag/llm.py`). 아래 문구는 규칙 식별을 위한 짧은 인용이다.

- `transform_query()`: “고유명사·문헌명·날짜·번호는 그대로 보존”, 원문에 없는 조건을 추가하지
  않고 서로 다른 사실을 물을 때만 최대 3개로 분해한다.
- `rerank()`: “질문의 답 또는 답의 일부가 명시된 문서만 yes”, 주제·용어만 비슷하거나
  애매하면 `no`로 판정한다. 결과는 문서 ID, `yes/no`, 원문 그대로의 근거를 포함한 JSON이다.
- `generate()`: “컨텍스트 밖의 지식·수치·추측은 금지”, 한 문장에 한 사실만 쓰고 모든 사실에
  출처 ID와 원문 인용을 붙인다. 근거가 없으면 “제공된 문서에서 확인되지 않는다.”라고 답한다.
- `verify_semantics()`: 초안을 원자적 주장으로 나누고 “사료에 명시된 경우에만” 지원된 것으로
  판정한다. 문서에 없는 조건, 과장, 상식에만 기댄 주장은 수정본에서 제거한다.

Gemini도 같은 변환·리랭킹 메서드를 위임받아 동일한 JSON 구조를 사용한다. GPT 클라이언트는
JSON 객체 출력을 요구하고, Gemini 클라이언트는 JSON MIME 형식으로 묶인다.

## Python 코드가 실제로 강제하는 조건

프롬프트 요청과 달리 다음 항목은 모델의 자율 판단에 맡기지 않고 코드가 검사한다.

- 질문이 비어 있으면 즉시 오류로 처리한다 (`AdvancedRAGPipeline.run`).
- 각 변환 질의의 검색 결과는 문서 ID로 합치며, 같은 ID는 점수가 높은 결과만 유지한다.
- 리랭킹 후보는 검색 점수순 상위 `max(final_k, 5)`개다.
- 리랭킹 결과의 ID가 실제 후보 ID와 같고, 판정이 `yes`이며, 근거 문자열이 본문에 그대로
  존재해야 통과한다. 생략 부호로 나눈 복수 구간도 모든 구간이 본문에 있어야 한다
  (`_evidence_is_verbatim`).
- 리랭킹 통과 문서가 없으면 생성 모델을 부르지 않고 `insufficient_evidence`를 반환한다.
- 생성 인용은 출처 ID 존재, 빈 인용 여부, 본문과의 정확한 문자열 일치를 검사한다
  (`AdvancedRAGPipeline._verify`).
- 의미 검증의 `answerable`, 유효 인용, 제거 사유·미지원 주장이 모두 충족되어야 `passed`다.
  그렇지 않으면 `needs_review` 경고를 붙인다.
- 검색·도구 실행 내역은 `PipelineResult.trace`에 남긴다.

검증 상태는 현재 `passed`, `needs_review`, `insufficient_evidence` 세 가지가 사용된다.

## 검색 도구와 출처 처리

`HybridHistoryRetriever`는 두 도구를 등록한다 (`src/history_rag/tool_calling.py`).

- `search_pinecone_history`: 14종 통합 코퍼스를 검색하며 `top_k`는 1~20으로 제한한다.
- `search_official_history`: 국사편찬위원회 조선왕조실록 검색·상세 페이지를 실시간 조회하며
  최대 3건만 반환한다. 조선왕조실록 범위가 아닌 검색어는 실행하지 않는다.

도구 호출은 한 검색당 최대 2회다. Gemini가 Pinecone 도구를 요청하지 않아도 Python이 한 번
대체 실행하여 기본 검색 경로를 보장한다. 결과는 기사 ID 기준으로 중복 제거하며, 공식 사이트
장애는 빈 결과와 오류 흔적으로 기록하고 Pinecone 검색을 중단시키지 않는다. 실시간 결과는
Pinecone에 저장하지 않는다 (`src/history_rag/official_search.py`).

`SourceDocument`는 문서 ID, 본문, 제목, 출처 URL, 메타데이터, 검색 점수를 보존한다
(`src/history_rag/models.py`). Pinecone 문서는 `source_url_type`으로 개별 기사 URL과 데이터셋
소개 URL을 구분하므로, 소개 페이지를 직접 인용 가능한 원문 링크처럼 표시해서는 안 된다.

## 이용권한과 외부 전송

외부 임베딩·Pinecone 적재 전 `validate_external_use()`와 `validate_rights_registry()`가 다음을
확인한다 (`src/history_rag/rights.py`).

- 필수 권한 메타데이터 존재
- `conditional_allowed` 및 AI 학습용 `explicit` 표시
- 공공데이터포털 상세 URL과 권한 근거 URL 일치
- 검토 완료 권한 대장의 조건부 허용·명시적 AI 허용·공식 URL 일치

출처 표시는 제공기관, 데이터셋명, 이용조건, 공식 URL을 포함한다. 이 판단은 특정 외부 업체를
명시한 개별 허가가 아니라 확인된 공개 이용조건에 따른 조건부 판단이라는 한계가 있다.

## 한계와 검증 범위

- 공식 실시간 검색 연결은 조선왕조실록 한 곳이다.
- 14종 코퍼스가 모든 시대·사건을 균등하게 포함하지는 않는다.
- LLM의 의미 판정은 확률적이지만, ID·인용 문자열·권한 메타데이터는 코드로 별도 검사한다.
- 외부 API, 공식 사이트, Pinecone의 장애·요금·사용량 제한에 영향을 받는다.
- 현재 회귀 검사는 `tests/`의 단위 테스트와 Ruff·Pyright 검사로 수행한다.


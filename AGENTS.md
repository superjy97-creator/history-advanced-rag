# Codex 유지보수 지침

이 파일은 앞으로 이 저장소를 다루는 Codex/AI 개발자를 위한 규칙이다. 과거 개발 전 과정에서
이 파일이 자동 적용되었다는 뜻은 아니다. 기존 구현에서 확인된 원칙을 정리하고 이후 변경에
적용할 유지보수 기준을 명시한다.

## 프로젝트 불변 조건

- Advanced RAG 순서는 질의 변환 → 검색 → 리랭킹 → 생성 → 검증이다.
- 모델 역할은 Gemini(질의 변환·리랭킹), GPT(생성·검증)로 유지한다.
- Pinecone은 Dense 3,072차원 cosine 인덱스 `history-advanced-rag`, namespace
  `korean-history-v1`을 사용한다.
- 기존 68,150개 벡터를 명시적 승인 없이 재임베딩·재적재·삭제하지 않는다.
- 원본 사료, 최종 JSONL, 권한 대장과 검증 결과를 임의로 수정하지 않는다.
- API 키, `.env`, Streamlit Secrets를 코드·로그·문서·Git에 기록하지 않는다.

## 변경 원칙

- 정상 작동하는 모델 프롬프트와 `llm.py`는 필요한 경우에만 최소 변경한다.
- 새 기능보다 기존 파이프라인의 근거성, 출처 추적, 회귀 방지를 우선한다.
- 데이터셋 소개 URL과 개별 원문 URL을 구분하고, 소개 URL을 직접 인용 링크로 표시하지 않는다.
- 관련 문서가 없거나 검증을 통과하지 못하면 근거 부족으로 처리한다. 리랭킹을 우회하거나
  무조건 상위 문서를 통과시키지 않는다.
- 외부 API 호출, 유료 작업, 데이터 외부 전송, Pinecone 리소스 변경은 사용자 승인을 받는다.
- 기존 파일을 삭제하거나 덮어쓰기 전에 대상과 영향 범위를 확인한다.

## 작업과 검증

1. 변경 전 관련 코드와 테스트, `git diff`를 확인한다.
2. 사용자 요청 범위 안에서 가장 작은 변경을 한다.
3. 최소한 아래 검사를 실행한다.

```powershell
C:\lab3\.venv\Scripts\python.exe -m ruff check src scripts tests app.py
C:\lab3\.venv\Scripts\python.exe -m pyright
C:\lab3\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

4. 외부 API를 쓰지 않는 단위 테스트와 실제 API 검증을 구분하여 보고한다.
5. 완료한 항목, 실제 검사 결과, 남은 문제를 구분하고 검증하지 않은 기능을 완료로 표시하지
   않는다.

자세한 모델·검증 규칙은 `docs/AI_답변_및_검증_지침.md`, 과제 요구사항과 구현 범위는
`docs/Advanced_RAG_과제_지침.md`를 참조한다.


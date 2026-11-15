from __future__ import annotations

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from history_rag.config import Settings
from history_rag.llm import GeminiHistoryLLM, OpenAIHistoryLLM, RoutedHistoryLLM
from history_rag.pipeline import AdvancedRAGPipeline
from history_rag.rights import attribution_text
from history_rag.tool_calling import HybridHistoryRetriever
from history_rag.vectorstore import PineconeHistoryStore

st.set_page_config(page_title="AI 역사 사료 해설", page_icon="📜", layout="wide")
st.title("📜 AI 역사 사료 해설")
st.caption("질의 변환 → Pinecone 검색 → 리랭킹 → 생성 → 검증")


@st.cache_resource
def build_pipeline() -> AdvancedRAGPipeline:
    settings = Settings.from_env()
    settings.validate_rag()
    llm = RoutedHistoryLLM(
        gemini=GeminiHistoryLLM(settings.gemini_api_key, settings.gemini_model),
        gpt=OpenAIHistoryLLM(settings.openai_api_key, settings.chat_model),
    )
    store = PineconeHistoryStore(
        settings.pinecone_api_key,
        settings.pinecone_index_name,
        settings.pinecone_namespace,
        settings.openai_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )
    tool_model = ChatGoogleGenerativeAI(
        google_api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )
    retriever = HybridHistoryRetriever(tool_model, store)
    return AdvancedRAGPipeline(llm, retriever)


question = st.text_input(
    "한국사 질문", placeholder="예: 신라 건국 설화에서 혁거세는 어떻게 묘사되나요?"
)
if st.button("사료로 해설하기", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("사료를 찾고 답변을 검증하고 있습니다..."):
            result = build_pipeline().run(question)
        st.subheader("검증된 해설")
        st.write(result.verification.final_answer)
        if result.verification.status == "passed":
            st.success("인용문과 출처가 원문에서 확인되었습니다.")
        else:
            st.warning("일부 근거는 추가 검토가 필요합니다.")
            for issue in result.verification.issues:
                st.write(f"- {issue}")

        st.subheader("출처와 인용 근거")
        sources = {d.id: d for d in result.reranked}
        for citation in result.verification.supported_citations:
            source = sources[citation.source_id]
            with st.expander(f"{source.title} · {citation.source_id}"):
                st.info(citation.quote)
                st.caption(attribution_text(source))
                link_label = (
                    "공식 사료 원문"
                    if source.metadata.get("source_url_type") == "official_record"
                    else "데이터셋 소개 페이지 (개별 원문 URL 아님)"
                )
                st.markdown(f"[{link_label}]({source.source_url})")
                st.json(source.metadata)

        with st.expander("4단계 실행 기록"):
            st.json(result.trace)
            st.write("변환된 검색 질의:", result.query.search_query)
    except Exception as exc:
        st.error(f"실행 중 오류가 발생했습니다: {exc}")

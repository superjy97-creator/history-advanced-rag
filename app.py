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

st.markdown(
    """
    <style>
    :root {
        --archive-navy: #12243a;
        --archive-navy-soft: #203952;
        --archive-gold: #b38a3e;
        --archive-gold-soft: #d7bd7b;
        --archive-paper: #f4ecd9;
        --archive-paper-deep: #e8dcc1;
        --archive-ink: #27241f;
        --archive-muted: #6f675b;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(179, 138, 62, 0.10), transparent 28rem),
            linear-gradient(180deg, #f8f2e5 0%, var(--archive-paper) 48%, #eee3cd 100%);
        color: var(--archive-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 236, 217, 0.88);
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .archive-hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.6rem 2.2rem;
        border: 1px solid rgba(179, 138, 62, 0.65);
        border-radius: 4px;
        background: linear-gradient(135deg, var(--archive-navy) 0%, #172f49 70%, #263f55 100%);
        box-shadow: 0 14px 35px rgba(24, 35, 48, 0.16);
        color: #fffaf0;
    }

    .archive-hero::after {
        content: "史";
        position: absolute;
        right: 1.8rem;
        top: -2.2rem;
        color: rgba(215, 189, 123, 0.10);
        font-family: Georgia, "Noto Serif KR", serif;
        font-size: 11rem;
        line-height: 1;
    }

    .archive-kicker {
        margin-bottom: 0.7rem;
        color: var(--archive-gold-soft);
        font-family: Georgia, serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.22em;
    }

    .archive-hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #fffaf0;
        font-family: Georgia, "Noto Serif KR", serif;
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 650;
        letter-spacing: -0.03em;
    }

    .archive-hero p {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 0.9rem 0 0;
        color: #e9dfca;
        font-size: 1rem;
        line-height: 1.75;
    }

    .archive-rule {
        width: 74px;
        height: 3px;
        margin-top: 1.3rem;
        background: var(--archive-gold);
    }

    .archive-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 1rem 0 2rem;
        border: 1px solid #d6c7a8;
        background: rgba(255, 252, 244, 0.72);
    }

    .archive-stat {
        padding: 1rem 1.2rem;
        border-right: 1px solid #d6c7a8;
    }

    .archive-stat:last-child { border-right: 0; }
    .archive-stat strong {
        display: block;
        color: var(--archive-navy);
        font-family: Georgia, "Noto Serif KR", serif;
        font-size: 1.35rem;
    }
    .archive-stat span {
        color: var(--archive-muted);
        font-size: 0.78rem;
        letter-spacing: 0.06em;
    }

    .section-eyebrow {
        margin-bottom: 0.25rem;
        color: #8e692b;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.18em;
    }

    .section-title {
        margin: 0 0 0.35rem;
        color: var(--archive-navy);
        font-family: Georgia, "Noto Serif KR", serif;
        font-size: 1.55rem;
    }

    .section-copy {
        margin: 0 0 1rem;
        color: var(--archive-muted);
        font-size: 0.92rem;
    }

    .research-steps {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 1.4rem 0 2rem;
    }

    .research-step {
        min-height: 118px;
        padding: 1rem;
        border-top: 3px solid var(--archive-gold);
        background: rgba(255, 252, 244, 0.78);
        box-shadow: 0 5px 16px rgba(54, 47, 35, 0.07);
    }

    .research-step b {
        display: block;
        margin-bottom: 0.35rem;
        color: var(--archive-navy);
        font-family: Georgia, "Noto Serif KR", serif;
    }

    .research-step span {
        color: var(--archive-muted);
        font-size: 0.79rem;
        line-height: 1.55;
    }

    .stTextInput input {
        border: 1px solid #bba77d;
        border-radius: 3px;
        background: #fffdf7;
        color: var(--archive-ink);
        min-height: 3.25rem;
    }

    .stButton > button[kind="primary"] {
        width: 100%;
        min-height: 3rem;
        border: 1px solid #8b672b;
        border-radius: 3px;
        background: var(--archive-navy);
        color: #fffaf0;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .stButton > button[kind="primary"]:hover {
        border-color: var(--archive-gold);
        background: var(--archive-navy-soft);
        color: #ffffff;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid #cdbb94;
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: var(--archive-navy);
        font-weight: 700;
    }

    [data-testid="stExpander"] {
        border-color: #d4c39f;
        background: rgba(255, 253, 247, 0.76);
    }

    @media (max-width: 760px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding: 1rem 1rem 3rem;
        }
        .archive-hero { padding: 1.6rem 1.25rem; }
        .archive-hero::after { display: none; }
        .archive-stats { grid-template-columns: 1fr; }
        .archive-stat {
            border-right: 0;
            border-bottom: 1px solid #d6c7a8;
        }
        .archive-stat:last-child { border-bottom: 0; }
        .research-steps { grid-template-columns: 1fr 1fr; }
    }

    @media (max-width: 440px) {
        .research-steps { grid-template-columns: 1fr; }
    }
    </style>

    <section class="archive-hero">
        <div class="archive-kicker">DIGITAL HISTORY LAB</div>
        <h1>AI 역사 사료 해설</h1>
        <p>고대부터 현대까지, 실제 사료를 검색하고 원문 인용과 출처를 검증하여
        연구에 활용할 수 있는 한국어 해설을 제공합니다.</p>
        <div class="archive-rule"></div>
    </section>

    <section class="archive-stats" aria-label="연구 아카이브 현황">
        <div class="archive-stat"><strong>14종</strong><span>HISTORICAL COLLECTIONS</span></div>
        <div class="archive-stat"><strong>68,150</strong><span>VERIFIED SEARCH CHUNKS</span></div>
        <div class="archive-stat"><strong>4단계</strong><span>ADVANCED RAG REVIEW</span></div>
    </section>

    <div class="section-eyebrow">RESEARCH ARCHIVE</div>
    <h2 class="section-title">사료 기반 질문 탐색</h2>
    <p class="section-copy">
        질문을 입력하면 검색부터 근거 검증까지 네 연구 단계를 순서대로 실행합니다.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="research-steps" aria-label="Advanced RAG 연구 단계">
        <div class="research-step">
            <b>01 · 질의 변환</b>
            <span>인물·시대·사건을 분석해 사료 검색어를 보완합니다.</span>
        </div>
        <div class="research-step">
            <b>02 · 사료 선별</b>
            <span>Pinecone 검색 결과에서 질문과 직접 관련된 근거만 고릅니다.</span>
        </div>
        <div class="research-step">
            <b>03 · 근거 해설</b>
            <span>선별된 원문만 사용해 인용이 연결된 해설을 작성합니다.</span>
        </div>
        <div class="research-step">
            <b>04 · 교차 검증</b>
            <span>주장·인용·출처를 다시 대조하고 검증 상태를 표시합니다.</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


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
    "한국사 질문",
    placeholder="예: 신라 건국 설화에서 혁거세는 어떻게 묘사되나요?",
)
if st.button("사료 검색 및 해설 시작", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("사료를 찾고 답변을 검증하고 있습니다..."):
            result = build_pipeline().run(question)
        st.markdown("---")
        answer_tab, source_tab, verification_tab = st.tabs(
            ["검증된 해설", "출처·인용", "검증 기록"]
        )

        with answer_tab:
            st.subheader("사료 기반 해설")
            st.write(result.verification.final_answer)
            if result.verification.status == "passed":
                st.success("인용문과 출처가 원문에서 확인되었습니다.")
            else:
                st.warning("일부 근거는 추가 검토가 필요합니다.")
                for issue in result.verification.issues:
                    st.write(f"- {issue}")

        with source_tab:
            st.subheader("출처와 직접 인용")
            sources = {d.id: d for d in result.reranked}
            if not result.verification.supported_citations:
                st.info("검증을 통과한 직접 인용이 없습니다.")
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

        with verification_tab:
            st.subheader("Advanced RAG 실행 기록")
            st.write("검증 상태:", result.verification.status)
            st.write("변환된 검색 질의:", result.query.search_query)
            st.json(result.trace)
    except Exception as exc:
        st.error(f"실행 중 오류가 발생했습니다: {exc}")

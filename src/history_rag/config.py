from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str = "korean-history-v1"
    chat_model: str = "gpt-6-luna"
    gemini_api_key: str = ""
    gemini_model: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT.parent / ".env", override=False)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "history-advanced-rag"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "korean-history-v1"),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", ""),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )

    def validate_live(self) -> None:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.pinecone_index_name:
            missing.append("PINECONE_INDEX_NAME")
        if missing:
            raise RuntimeError("필수 환경변수가 없습니다: " + ", ".join(missing))
        expected_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        expected = expected_dimensions.get(self.embedding_model)
        if expected is not None and self.embedding_dimension != expected:
            raise RuntimeError(
                f"{self.embedding_model} 기본 차원은 {expected}인데 "
                f"설정은 {self.embedding_dimension}입니다."
            )

    def validate_rag(self) -> None:
        self.validate_live()
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.gemini_model:
            missing.append("GEMINI_MODEL")
        if not self.chat_model:
            missing.append("OPENAI_CHAT_MODEL")
        if missing:
            raise RuntimeError("RAG model configuration is missing: " + ", ".join(missing))

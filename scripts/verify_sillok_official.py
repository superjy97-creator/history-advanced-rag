from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

TARGET_CHARACTER = "𥘼"


def official_article_id(article_id: str) -> str:
    if not re.fullmatch(r"w[qr][ab]_\d{8}_\d{3}", article_id):
        raise ValueError(f"지원하지 않는 실록 기사 ID: {article_id}")
    return "k" + article_id[1:]


def visible_text(raw_html: str) -> str:
    without_scripts = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", "", html.unescape(without_tags))


def contexts(text: str, radius: int = 20) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    return [
        compact[max(0, index - radius) : index + radius + 1]
        for index, character in enumerate(compact)
        if character == TARGET_CHARACTER
    ]


def load_targets(path: Path) -> list[dict]:
    targets = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if TARGET_CHARACTER in row["text"]:
                targets.append(row)
    return targets


def fetch(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 history-source-verification/1.0"})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return response.read().decode("utf-8", errors="strict")


def verify(input_path: Path, output_path: Path, request_interval: float = 0.25) -> dict:
    if output_path.exists():
        raise FileExistsError(f"기존 보고서는 덮어쓰지 않습니다: {output_path}")
    results = []
    targets = load_targets(input_path)
    for position, row in enumerate(targets):
        article_id = str(row["metadata"]["article_id"])
        official_id = official_article_id(article_id)
        official_url = f"https://sillok.history.go.kr/id/{official_id}"
        local_contexts = contexts(row["text"])
        result = {
            "article_id": article_id,
            "official_article_id": official_id,
            "date": row["metadata"].get("date", ""),
            "title": row["title"],
            "xml_file": row["metadata"].get("xml_file", ""),
            "official_url": official_url,
            "problem_serialization": "&#55381;&#56892;",
            "restored_character": TARGET_CHARACTER,
            "unicode_codepoint": "U+2563C",
            "occurrences": len(local_contexts),
            "local_contexts": local_contexts,
            "matched_contexts": [],
            "status": "unverified",
            "reason": "",
        }
        try:
            raw_html = fetch(official_url)
            official_text = visible_text(raw_html)
            matched = [context in official_text for context in local_contexts]
            result["matched_contexts"] = matched
            result["official_page_sha256"] = hashlib.sha256(
                raw_html.encode("utf-8")
            ).hexdigest()
            result["official_character_occurrences"] = official_text.count(TARGET_CHARACTER)
            title_match = re.sub(r"\s+", "", row["title"]) in official_text
            result["title_match"] = title_match
            year, month, day = str(row["metadata"].get("date", "")).split("-")
            date_match = (
                f"{int(year)}년" in official_text
                and f"{int(month)}월" in official_text
                and f"{int(day)}일" in official_text
            )
            result["date_match"] = date_match
            result["article_id_match"] = official_id in raw_html
            if all(matched) and date_match and result["article_id_match"]:
                result["status"] = "verified_official_match"
                result["reason"] = "기사 ID·날짜와 모든 복원 문자 문맥이 공식 원문에 일치"
            else:
                result["status"] = "mismatch"
                result["reason"] = "기사 ID·날짜 또는 복원 문자 문맥이 공식 원문과 불일치"
        except Exception as exc:  # noqa: BLE001 - audit must record every access failure
            result["reason"] = f"공식 페이지 확인 실패: {type(exc).__name__}: {exc}"
        results.append(result)
        if position + 1 < len(targets):
            time.sleep(request_interval)

    summary = {
        "checked_at": "2026-11-14",
        "source_file": str(input_path),
        "documents": len(results),
        "occurrences": sum(item["occurrences"] for item in results),
        "verified": sum(item["status"] == "verified_official_match" for item in results),
        "mismatched": sum(item["status"] == "mismatch" for item in results),
        "unverified": sum(item["status"] == "unverified" for item in results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_fields = ("documents", "occurrences", "verified", "mismatched", "unverified")
    return {key: summary[key] for key in summary_fields}


def main() -> None:
    parser = argparse.ArgumentParser(description="실록 확장한자 복원을 공식 원문과 기사별 대조")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-interval", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(verify(args.input, args.output, args.request_interval), ensure_ascii=False))


if __name__ == "__main__":
    main()

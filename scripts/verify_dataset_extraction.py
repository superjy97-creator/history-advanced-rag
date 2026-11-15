from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from dataset_audit_extract import MARKER_NAME, recover_zip_name, safe_relative, sha256_file


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_zip(source: Path, target: Path) -> dict:
    errors: list[str] = []
    decisions: Counter[str] = Counter()
    expected_files: set[str] = set()
    with zipfile.ZipFile(source) as archive:
        bad_crc = archive.testzip()
        if bad_crc is not None:
            errors.append(f"CRC32 failure: {bad_crc}")
        for info in archive.infolist():
            recovered, decision = recover_zip_name(info)
            decisions[decision] += 1
            relative = safe_relative(recovered.rstrip("/"))
            if info.is_dir():
                if not (target / relative).is_dir():
                    errors.append(f"missing directory: {relative}")
                continue
            expected_files.add(relative.as_posix().casefold())
            extracted = target / relative
            if not extracted.is_file():
                errors.append(f"missing file: {relative}")
                continue
            raw = archive.read(info)
            if extracted.stat().st_size != info.file_size:
                errors.append(f"size mismatch: {relative}")
            elif sha256_file(extracted) != bytes_sha256(raw):
                errors.append(f"content mismatch: {relative}")

    actual_files = {
        path.relative_to(target).as_posix().casefold()
        for path in target.rglob("*")
        if path.is_file() and path.name != MARKER_NAME
    }
    extras = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    errors.extend(f"unexpected extracted file: {name}" for name in extras)
    errors.extend(f"missing extracted file: {name}" for name in missing)
    return {
        "status": "verified" if not errors else "problem",
        "file_count": len(expected_files),
        "filename_decisions": dict(decisions),
        "errors": errors,
    }


def verify_csv(source: Path, target: Path, audit_record: dict) -> dict:
    errors: list[str] = []
    extracted = target / source.name
    if not extracted.is_file():
        errors.append(f"missing copied CSV: {source.name}")
    elif sha256_file(extracted) != sha256_file(source):
        errors.append("copied CSV hash differs from source")

    row_count = 0
    widths: Counter[int] = Counter()
    encoding = audit_record["encoding"]
    with source.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        for row in reader:
            row_count += 1
            widths[len(row)] += 1
    if row_count != audit_record["data_row_count"]:
        errors.append("CSV row count differs from audit report")
    if len(header) != audit_record["column_count"]:
        errors.append("CSV column count differs from audit report")
    if any(width != len(header) for width in widths):
        errors.append(f"irregular CSV row widths: {dict(widths)}")
    return {
        "status": "verified" if not errors else "problem",
        "file_count": 1,
        "data_row_count": row_count,
        "column_count": len(header),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="원본과 안전 추출 결과를 바이트 단위로 재검증")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"기존 검증 보고서를 덮어쓰지 않습니다: {args.output}")

    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    results = []
    source_hashes: Counter[str] = Counter()
    report_names = {record["source_file"] for record in audit["datasets"]}
    inbox_names = {path.name for path in args.inbox.iterdir() if path.is_file()}
    inventory_errors = []
    if report_names != inbox_names:
        inventory_errors.append("audit report and inbox inventory differ")

    for record in audit["datasets"]:
        source = args.inbox / record["source_file"]
        target = args.raw_root / record["dataset"]
        errors: list[str] = []
        current_hash = sha256_file(source)
        source_hashes[current_hash] += 1
        if current_hash != record["sha256"]:
            errors.append("source SHA-256 differs from audit report")
        marker_path = target / MARKER_NAME
        if not marker_path.is_file():
            errors.append("missing extraction manifest")
        else:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("source_sha256") != current_hash:
                errors.append("manifest SHA-256 differs from source")
            if marker.get("status") != "complete":
                errors.append("manifest status is not complete")

        if source.suffix.lower() == ".zip":
            detail = verify_zip(source, target)
        else:
            detail = verify_csv(source, target, record)
        errors.extend(detail.pop("errors"))
        if marker_path.is_file():
            marker_count = json.loads(marker_path.read_text(encoding="utf-8")).get("file_count")
            if marker_count != detail["file_count"]:
                errors.append("manifest file count differs from extracted content")
        results.append(
            {
                "dataset": record["dataset"],
                "status": "verified" if not errors else "problem",
                **detail,
                "errors": errors,
            }
        )

    duplicate_groups = sum(count > 1 for count in source_hashes.values())
    output = {
        "source_count": len(results),
        "verified_count": sum(item["status"] == "verified" for item in results),
        "problem_count": sum(item["status"] == "problem" for item in results),
        "duplicate_sha256_groups": duplicate_groups,
        "inventory_errors": inventory_errors,
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "datasets"}, ensure_ascii=False
        )
    )
    if output["problem_count"] or inventory_errors or duplicate_groups:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

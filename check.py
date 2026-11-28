import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import kino


def _read_last_rejected_count() -> int:
    if not kino.CALLS_LOG.exists():
        return 0
    lines = kino.CALLS_LOG.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0
    last = json.loads(lines[-1])
    return len(last.get("rejected", []))


def main() -> None:
    load_dotenv()
    films = json.loads(Path("films.json").read_text(encoding="utf-8"))
    queries = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()

    outcomes: Counter[str] = Counter()
    caught_foreign_ids = 0
    expect_total = 0
    expect_passed = 0

    for idx, item in enumerate(queries, 1):
        query = item["query"]
        expect = item.get("expect")
        print(f"\n[{idx}/{len(queries)}] {query}")
        outcome, ids = kino.pick(client, model, query, films)
        outcomes[outcome] += 1
        caught_foreign_ids += _read_last_rejected_count()

        if expect:
            expect_total += 1
            passed = True
            if expect.get("no_match"):
                passed = outcome == "no_match"
            if "must" in expect:
                passed = passed and outcome == "ok" and all(m in ids for m in expect["must"])
            if "forbid" in expect:
                passed = passed and all(f not in ids for f in expect["forbid"])
            if passed:
                expect_passed += 1
            print(f"  expect: {'PASS' if passed else 'FAIL'} (outcome={outcome}, ids={ids})")

    print("\n=== Итог check.py ===")
    print(f"Всего запросов: {len(queries)}")
    print(f"ok: {outcomes['ok']}, no_match: {outcomes['no_match']}, invalid: {outcomes['invalid']}")
    other = {k: v for k, v in outcomes.items() if k not in {"ok", "no_match", "invalid"}}
    if other:
        print(f"Прочие исходы: {other}")
    print(f"Поймано чужих id: {caught_foreign_ids}")
    print(f"Запросов с ожиданием пройдено: {expect_passed}/{expect_total}")


if __name__ == "__main__":
    main()

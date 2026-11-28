import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import kino


def _read_last_log_stats() -> tuple[int, float]:
    if not kino.CALLS_LOG.exists():
        return 0, 0.0
    lines = kino.CALLS_LOG.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0, 0.0
    last = json.loads(lines[-1])
    return len(last.get("rejected", [])), float(last.get("cost", 0.0))


def run_once(client: OpenAI, model: str, films: list[dict], queries: list[dict]) -> dict:
    outcomes: Counter[str] = Counter()
    caught_foreign_ids = 0
    expect_total = 0
    expect_passed = 0
    total_cost = 0.0
    t0 = time.perf_counter()

    for idx, item in enumerate(queries, 1):
        query = item["query"]
        expect = item.get("expect")
        print(f"\n[{idx}/{len(queries)}] {query}")
        outcome, ids = kino.pick(client, model, query, films)
        outcomes[outcome] += 1
        rej, cost = _read_last_log_stats()
        caught_foreign_ids += rej
        total_cost += cost

        if expect:
            expect_total += 1
            if expect.get("no_match"):
                passed = outcome == "no_match"
            else:
                passed = outcome == "ok"
                if "must" in expect:
                    passed = passed and any(m in ids for m in expect["must"])
                if "forbid" in expect:
                    passed = passed and all(f not in ids for f in expect["forbid"])
            if passed:
                expect_passed += 1
            print(f"  expect: {'PASS' if passed else 'FAIL'} (outcome={outcome}, ids={ids})")

    elapsed = time.perf_counter() - t0
    pass_rate = (expect_passed / expect_total * 100) if expect_total else 0.0
    print("\n=== Итог check.py ===")
    print(f"Модель: {model}")
    print(f"Всего запросов: {len(queries)}")
    print(f"ok: {outcomes['ok']}, no_match: {outcomes['no_match']}, invalid: {outcomes['invalid']}")
    print(f"Поймано чужих/нарушающих id: {caught_foreign_ids}")
    print(f"Запросов с ожиданием пройдено: {expect_passed}/{expect_total} ({pass_rate:.1f}%)")
    print(f"Время: {elapsed:.1f} сек, стоимость (choice): ${total_cost:.6f}")

    return {
        "passed": expect_passed,
        "total": expect_total,
        "pass_rate": pass_rate,
        "rejected": caught_foreign_ids,
        "elapsed": elapsed,
        "cost": total_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон queries.json")
    parser.add_argument("--model", default=None, help="Имя модели (по умолчанию из .env)")
    parser.add_argument("--runs", type=int, default=1, help="Число прогонов")
    args = parser.parse_args()

    load_dotenv()
    films = json.loads(Path("films.json").read_text(encoding="utf-8"))
    queries = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()

    results = []
    for r in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n========== ПРОГОН {r}/{args.runs} ({model}) ==========")
        results.append(run_once(client, model, films, queries))

    if args.runs > 1:
        avg_rate = sum(x["pass_rate"] for x in results) / len(results)
        avg_time = sum(x["elapsed"] for x in results) / len(results)
        total_rej = sum(x["rejected"] for x in results)
        total_cost = sum(x["cost"] for x in results)
        print(f"\n=== Сводка за {args.runs} прогона ({model}) ===")
        print(f"Средняя доля пройденных: {avg_rate:.1f}%")
        print(f"Всего поймано чужих/нарушающих id: {total_rej}")
        print(f"Среднее время прогона: {avg_time:.1f} сек, суммарная стоимость: ${total_cost:.6f}")


if __name__ == "__main__":
    main()

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import kino


def count_tokens_approx(data: list[dict]) -> int:
    text = json.dumps(data, ensure_ascii=False)
    return len(text) // 4


def _read_last_log() -> dict:
    lines = kino.CALLS_LOG.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else {}


def main() -> None:
    load_dotenv()
    films_big = json.loads(Path("films_big.json").read_text(encoding="utf-8"))
    queries = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()

    big_tokens = count_tokens_approx(films_big)
    print(f"Каталог films_big.json: {len(films_big)} фильмов, ~{big_tokens} токенов.")

    # Замер 1: Recall@30 без языковой модели
    must_queries = [q for q in queries if "must" in q.get("expect", {})]
    recall_hits = 0
    for q in must_queries:
        cand_ids = kino.candidates(q["query"], films_big, k=30)
        if any(m in cand_ids for m in q["expect"]["must"]):
            recall_hits += 1
    print(f"Замер 1 (Recall@30): {recall_hits}/{len(must_queries)} ({recall_hits / len(must_queries) * 100:.1f}%)")

    # Замер 2: сравнение «30 кандидатов (pick_big)» vs «весь каталог (pick)»
    for mode_name, fn in [("30 кандидатов (pick_big)", kino.pick_big), ("Весь каталог (pick)", kino.pick)]:
        passed = 0
        total_tokens = 0
        total_cost = 0.0
        t0 = time.perf_counter()
        for q in queries:
            expect = q["expect"]
            outcome, ids = fn(client, model, q["query"], films_big)
            last = _read_last_log()
            total_tokens += last.get("tokens", {}).get("total", 0)
            total_cost += last.get("cost", 0.0)
            if expect.get("no_match"):
                ok = outcome == "no_match"
            else:
                ok = outcome == "ok" and any(m in ids for m in expect.get("must", [])) and all(
                    f not in ids for f in expect.get("forbid", [])
                )
            if ok:
                passed += 1
        elapsed = time.perf_counter() - t0
        print(
            f"Замер 2 [{mode_name}]: пройдено {passed}/{len(queries)} "
            f"({passed / len(queries) * 100:.1f}%), токены={total_tokens}, "
            f"цена=${total_cost:.6f}, время={elapsed:.1f} сек"
        )


if __name__ == "__main__":
    main()

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    log_path = Path("calls.jsonl")
    if not log_path.exists():
        print("Файл calls.jsonl не найден.")
        return

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        print("Журнал calls.jsonl пуст.")
        return

    outcomes: Counter[str] = Counter()
    steps: Counter[str] = Counter()
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    total_rejected = 0

    for line in lines:
        entry = json.loads(line)
        outcomes[entry.get("outcome", "unknown")] += 1
        steps[entry.get("step", "choice")] += 1
        tokens = entry.get("tokens", {})
        total_prompt += tokens.get("prompt", 0)
        total_completion += tokens.get("completion", 0)
        total_cost += entry.get("cost", 0.0)
        total_rejected += len(entry.get("rejected", []))

    total_calls = len(lines)
    total_tokens = total_prompt + total_completion
    avg_cost = total_cost / total_calls if total_calls else 0.0

    print("=== Итог show.py ===")
    print(f"Всего вызовов API: {total_calls} (шаги: {', '.join(f'{k}={v}' for k, v in steps.items())})")
    print("Исходы:", ", ".join(f"{k}: {v}" for k, v in outcomes.items()))
    print(f"Отклонённых id: {total_rejected}")
    print(f"Токены: prompt={total_prompt}, completion={total_completion}, total={total_tokens}")
    print(f"Общая стоимость: ${total_cost:.6f}")
    print(f"Средняя стоимость 1 вызова: ${avg_cost:.6f}")


if __name__ == "__main__":
    main()

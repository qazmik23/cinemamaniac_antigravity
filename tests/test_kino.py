import json
from types import SimpleNamespace

from openai import APIConnectionError, APIStatusError

import kino

SAMPLE_FILMS = [
    {
        "id": "tt0374089",
        "title": "Кочевник",
        "year": 2005,
        "minutes": 112,
        "genres": ["История", "Драма"],
        "countries": ["Казахстан"],
        "languages": ["kk", "ru"],
        "description": "Историческая драма о становлении Казахского ханства.",
    },
    {
        "id": "tt7985648",
        "title": "Томирис",
        "year": 2019,
        "minutes": 155,
        "genres": ["История", "Боевик"],
        "countries": ["Казахстан"],
        "languages": ["kk", "ru"],
        "description": "История сакской царицы Томирис.",
    },
    {
        "id": "tt1677719",
        "title": "Рэкетир",
        "year": 2007,
        "minutes": 80,
        "genres": ["Криминал", "Драма"],
        "countries": ["Казахстан"],
        "languages": ["ru"],
        "description": "Криминальная драма о лихих 90-х в Казахстане.",
    },
    {
        "id": "tt0035423",
        "title": "Кейт и Лео",
        "year": 2001,
        "minutes": 118,
        "genres": ["Комедия", "Мелодрама"],
        "countries": ["США"],
        "languages": ["ru", "en"],
        "description": "Романтическая комедия о путешествии во времени.",
    },
]


def _make_mock_client(parsed=None, refusal=None, finish_reason="stop", exc=None):
    def parse(**kwargs):
        if exc is not None:
            raise exc
        msg = SimpleNamespace(parsed=parsed, refusal=refusal, content=None)
        choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[choice], usage=usage)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))


def test_ok_pick(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    parsed = kino.RecommendationResponse(
        status="ok",
        reason="Подходит по запросу.",
        picks=[kino.PickItem(film_id="tt0374089", why="Исторический фильм.")],
    )
    client = _make_mock_client(parsed=parsed)
    outcome, ids = kino.pick(client, "gpt-4o-mini", "история", SAMPLE_FILMS)
    assert outcome == "ok"
    assert ids == ["tt0374089"]


def test_foreign_id_rejected(tmp_path, monkeypatch):
    log_file = tmp_path / "calls.jsonl"
    monkeypatch.setattr(kino, "CALLS_LOG", log_file)
    parsed = kino.RecommendationResponse(
        status="ok",
        reason="Выбор.",
        picks=[kino.PickItem(film_id="tt9999999", why="Выдуманный фильм.")],
    )
    client = _make_mock_client(parsed=parsed)
    outcome, ids = kino.pick(client, "gpt-4o-mini", "что-нибудь", SAMPLE_FILMS)
    assert outcome == "invalid"
    assert ids == []
    logged = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert logged["rejected"] == ["tt9999999"]


def test_duplicate_ids_deduplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    parsed = kino.RecommendationResponse(
        status="ok",
        reason="Повтор.",
        picks=[
            kino.PickItem(film_id="tt0374089", why="Раз."),
            kino.PickItem(film_id="tt0374089", why="Два."),
        ],
    )
    client = _make_mock_client(parsed=parsed)
    outcome, ids = kino.pick(client, "gpt-4o-mini", "история", SAMPLE_FILMS)
    assert outcome == "ok"
    assert ids == ["tt0374089"]


def test_more_than_three_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    parsed = kino.RecommendationResponse(
        status="ok",
        reason="Много.",
        picks=[
            kino.PickItem(film_id="tt0374089", why="1"),
            kino.PickItem(film_id="tt7985648", why="2"),
            kino.PickItem(film_id="tt1677719", why="3"),
            kino.PickItem(film_id="tt0035423", why="4"),
        ],
    )
    client = _make_mock_client(parsed=parsed)
    outcome, ids = kino.pick(client, "gpt-4o-mini", "всё", SAMPLE_FILMS)
    assert outcome == "ok"
    assert ids == ["tt0374089", "tt7985648", "tt1677719"]


def test_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    parsed = kino.RecommendationResponse(status="no_match", reason="Нет таких.", picks=[])
    client = _make_mock_client(parsed=parsed)
    outcome, ids = kino.pick(client, "gpt-4o-mini", "мультфильм 2031", SAMPLE_FILMS)
    assert outcome == "no_match"
    assert ids == []


def test_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    client = _make_mock_client(refusal="Не могу ответить.")
    outcome, ids = kino.pick(client, "gpt-4o-mini", "запрос", SAMPLE_FILMS)
    assert outcome == "refusal"
    assert ids == []


def test_length_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    client = _make_mock_client(finish_reason="length")
    outcome, ids = kino.pick(client, "gpt-4o-mini", "запрос", SAMPLE_FILMS)
    assert outcome == "length"
    assert ids == []


def test_no_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    req = SimpleNamespace(url="https://api.openai.com/v1/chat/completions", method="POST", headers={})
    client = _make_mock_client(exc=APIConnectionError(request=req))
    outcome, ids = kino.pick(client, "gpt-4o-mini", "запрос", SAMPLE_FILMS)
    assert outcome == "no_connection"
    assert ids == []


def test_api_status_error(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")
    req = SimpleNamespace(url="https://api.openai.com/v1/chat/completions", method="POST", headers={})
    resp = SimpleNamespace(status_code=500, request=req, headers={}, history=[], next_request=None)
    client = _make_mock_client(exc=APIStatusError("Server error", response=resp, body=None))
    outcome, ids = kino.pick(client, "gpt-4o-mini", "запрос", SAMPLE_FILMS)
    assert outcome == "error_500"
    assert ids == []


def test_missing_why_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "CALLS_LOG", tmp_path / "calls.jsonl")

    def parse(**kwargs):
        raw = json.dumps({"status": "ok", "reason": "Ок", "picks": [{"film_id": "tt0374089"}]})
        msg = SimpleNamespace(parsed=None, refusal=None, content=raw)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    outcome, ids = kino.pick(client, "gpt-4o-mini", "история", SAMPLE_FILMS)
    assert outcome == "invalid"
    assert ids == []

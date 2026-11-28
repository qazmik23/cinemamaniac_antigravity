import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GENRE_MAP = {
    "Action": "Боевик",
    "Adventure": "Приключения",
    "Animation": "Мультфильм",
    "Biography": "Биография",
    "Comedy": "Комедия",
    "Crime": "Криминал",
    "Documentary": "Документальный",
    "Drama": "Драма",
    "Family": "Семейный",
    "Fantasy": "Фэнтези",
    "Film-Noir": "Нуар",
    "History": "История",
    "Horror": "Ужасы",
    "Music": "Музыка",
    "Musical": "Мюзикл",
    "Mystery": "Детектив",
    "News": "Новости",
    "Romance": "Мелодрама",
    "Sci-Fi": "Фантастика",
    "Short": "Короткометражка",
    "Sport": "Спорт",
    "Thriller": "Триллер",
    "War": "Военный",
    "Western": "Вестерн",
}

COUNTRY_MAP = {
    "KZ": "Казахстан",
    "RU": "Россия",
    "SU": "СССР",
    "US": "США",
    "GB": "Великобритания",
    "FR": "Франция",
    "DE": "Германия",
    "IT": "Италия",
    "ES": "Испания",
    "CA": "Канада",
    "AU": "Австралия",
    "JP": "Япония",
    "KR": "Южная Корея",
    "CN": "Китай",
    "HK": "Гонконг",
    "IN": "Индия",
    "MX": "Мексика",
    "BR": "Бразилия",
    "AR": "Аргентина",
    "SE": "Швеция",
    "NO": "Норвегия",
    "DK": "Дания",
    "FI": "Финляндия",
    "PL": "Польша",
    "CZ": "Чехия",
    "NL": "Нидерланды",
    "BE": "Бельгия",
    "CH": "Швейцария",
    "AT": "Австрия",
    "TR": "Турция",
    "IR": "Иран",
    "IL": "Израиль",
    "UA": "Украина",
    "BY": "Беларусь",
    "UZ": "Узбекистан",
    "KG": "Кыргызстан",
    "AZ": "Азербайджан",
    "GE": "Грузия",
    "AM": "Армения",
}


def load_env(path: str = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tmdb_get(endpoint: str, params: dict[str, str] | None = None) -> dict:
    api_key = os.environ.get("TMDB_API_KEY")
    read_token = os.environ.get("TMDB_READ_TOKEN")
    if not api_key and not read_token:
        raise RuntimeError("В .env не найден TMDB_API_KEY или TMDB_READ_TOKEN")

    query = dict(params or {})
    headers = {"Accept": "application/json"}
    if read_token:
        headers["Authorization"] = f"Bearer {read_token}"
    else:
        query["api_key"] = api_key

    url = f"https://api.themoviedb.org/3{endpoint}?{urllib.parse.urlencode(query)}"
    for attempt in range(4):
        time.sleep(0.12)
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                retry_after = int(e.headers.get("Retry-After", 2 ** (attempt + 1)))
                time.sleep(retry_after)
                continue
            raise
    raise RuntimeError(f"TMDB не ответил после повторов: {endpoint}")


def translate_overview_openai(text: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not text.strip():
        return ""

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Переведи описание фильма на русский язык. "
                    "Выход строго на русском языке, не длиннее 4 предложений, передай только основную идею."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def collect_candidates(tsv_path: str = "title.basics.tsv", limit: int = 400) -> list[dict]:
    candidates = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < 9:
                continue
            tconst, title_type, primary_title, _, is_adult, start_year, _, runtime, genres = cols
            if (
                title_type == "movie"
                and is_adult == "0"
                and start_year.isdigit()
                and int(start_year) > 1990
                and runtime.isdigit()
                and 60 < int(runtime) <= 140
                and genres != r"\N"
            ):
                candidates.append(
                    {
                        "id": tconst,
                        "primary_title": primary_title,
                        "year": int(start_year),
                        "minutes": int(runtime),
                        "genres": [GENRE_MAP.get(g, g) for g in genres.split(",")],
                    }
                )
                if len(candidates) >= limit:
                    break
    return candidates


def load_akas_for_ids(target_ids: set[str], akas_path: str = "title.akas.tsv") -> dict[str, dict]:
    akas_info: dict[str, dict] = {
        tid: {"ru_title": "", "kk_title": "", "languages": set()} for tid in target_ids
    }
    with open(akas_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < 8:
                continue
            tid, _, title, region, language = cols[0], cols[1], cols[2], cols[3], cols[4]
            if tid not in akas_info:
                continue
            info = akas_info[tid]
            if language != r"\N" and language:
                info["languages"].add(language)
            elif region != r"\N" and region:
                info["languages"].add(region)

            if (language == "ru" or region == "RU") and not info["ru_title"]:
                info["ru_title"] = title
            if (language == "kk" or region == "KZ") and not info["kk_title"]:
                info["kk_title"] = title
    return akas_info


def map_countries(prod_countries: list[dict]) -> list[str]:
    result = []
    for c in prod_countries:
        iso = c.get("iso_3166_1", "")
        name = COUNTRY_MAP.get(iso) or c.get("name", "")
        if name and name not in result:
            result.append(name)
    return result


def get_ru_description(tmdb_id: int, ru_overview: str) -> str:
    if ru_overview and ru_overview.strip():
        return ru_overview.strip()
    en_details = tmdb_get(f"/movie/{tmdb_id}", {"language": "en-US"})
    en_overview = en_details.get("overview", "").strip()
    if not en_overview:
        return ""
    return translate_overview_openai(en_overview)


def build_record(
    film_id: str,
    title: str,
    year: int,
    minutes: int,
    genres: list[str],
    countries: list[str],
    languages: list[str],
    description: str,
    source: str,
) -> dict | None:
    if not languages and "Казахстан" in countries:
        languages = ["Казахстан"]
    if not any(lang in ("ru", "RU", "kk", "KZ", "Казахстан") for lang in languages):
        return None
    if not all([film_id, title, year, minutes, genres, countries, languages, description, source]):
        return None
    return {
        "id": film_id,
        "title": title,
        "year": year,
        "minutes": minutes,
        "genres": genres,
        "countries": countries,
        "languages": languages,
        "description": description,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор каталога films.json")
    parser.add_argument("--count", type=int, default=40, help="Количество фильмов из title.basics.tsv")
    parser.add_argument("--kz-count", type=int, default=12, help="Количество казахстанских фильмов из TMDB")
    parser.add_argument("--out", default="films.json", help="Имя выходного JSON-файла")
    args = parser.parse_args()

    load_env()
    print("1. Отбор кандидатов из title.basics.tsv...")
    candidates = collect_candidates(limit=max(400, args.count * 10))
    akas = load_akas_for_ids({c["id"] for c in candidates})

    films: list[dict] = []
    seen_ids: set[str] = set()

    print(f"2. Обогащение {args.count} фильмов через TMDB и OpenAI...")
    for cand in candidates:
        if len(films) >= args.count:
            break
        tconst = cand["id"]
        try:
            find_data = tmdb_get(f"/find/{tconst}", {"external_source": "imdb_id", "language": "ru-RU"})
            results = find_data.get("movie_results", [])
            if not results:
                continue
            tmdb_id = results[0]["id"]
            details = tmdb_get(f"/movie/{tmdb_id}", {"language": "ru-RU"})
        except Exception:
            continue

        countries = map_countries(details.get("production_countries", []))
        akas_entry = akas.get(tconst, {})
        if "Казахстан" in countries:
            title = akas_entry.get("kk_title") or akas_entry.get("ru_title") or details.get("title", "")
        else:
            title = akas_entry.get("ru_title") or details.get("title", "")

        languages = sorted(akas_entry.get("languages", set()))
        if not languages:
            languages = [
                lang["iso_639_1"]
                for lang in details.get("spoken_languages", [])
                if lang.get("iso_639_1")
            ]

        description = get_ru_description(tmdb_id, details.get("overview", ""))
        record = build_record(
            film_id=tconst,
            title=title,
            year=cand["year"],
            minutes=cand["minutes"],
            genres=cand["genres"],
            countries=countries,
            languages=languages,
            description=description,
            source=f"https://www.imdb.com/title/{tconst}",
        )
        if record:
            films.append(record)
            seen_ids.add(tconst)
            print(f"  [{len(films)}/{args.count}] {record['id']} — {record['title']}")

    print(f"3. Добор {args.kz_count} фильмов, снятых в Казахстане, из TMDB...")
    kz_added = 0
    page = 1
    while kz_added < args.kz_count and page <= 5:
        discover = tmdb_get(
            "/discover/movie",
            {
                "with_origin_country": "KZ",
                "language": "ru-RU",
                "sort_by": "popularity.desc",
                "page": str(page),
            },
        )
        for item in discover.get("results", []):
            if kz_added >= args.kz_count:
                break
            tmdb_id = item["id"]
            try:
                details = tmdb_get(
                    f"/movie/{tmdb_id}",
                    {"language": "ru-RU", "append_to_response": "external_ids"},
                )
            except Exception:
                continue

            imdb_id = (details.get("external_ids") or {}).get("imdb_id") or details.get("imdb_id")
            film_id = imdb_id if imdb_id else f"tmdb-{tmdb_id}"
            if film_id in seen_ids:
                continue

            release_date = details.get("release_date", "")
            year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else 0
            minutes = details.get("runtime") or 0
            genres = [g["name"].capitalize() for g in details.get("genres", []) if g.get("name")]
            countries = map_countries(details.get("production_countries", []))
            if "Казахстан" not in countries:
                countries.insert(0, "Казахстан")
            languages = [
                lang["iso_639_1"]
                for lang in details.get("spoken_languages", [])
                if lang.get("iso_639_1")
            ]
            description = get_ru_description(tmdb_id, details.get("overview", ""))
            source = (
                f"https://www.imdb.com/title/{imdb_id}"
                if imdb_id
                else f"https://www.themoviedb.org/movie/{tmdb_id}"
            )

            record = build_record(
                film_id=film_id,
                title=details.get("title", ""),
                year=year,
                minutes=minutes,
                genres=genres,
                countries=countries,
                languages=languages,
                description=description,
                source=source,
            )
            if record:
                films.append(record)
                seen_ids.add(film_id)
                kz_added += 1
                print(f"  [KZ {kz_added}/12] {record['id']} — {record['title']}")
        page += 1

    Path(args.out).write_text(json.dumps(films, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово! Сохранено {len(films)} фильмов в {args.out}.")


if __name__ == "__main__":
    main()

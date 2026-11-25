import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


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
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_first_movie(tsv_path: str = "title.basics.tsv", min_year: int = 1990) -> dict:
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (
                row["titleType"] == "movie"
                and row["isAdult"] == "0"
                and row["startYear"].isdigit()
                and int(row["startYear"]) >= min_year
                and row["runtimeMinutes"].isdigit()
            ):
                return row
    raise RuntimeError("Подходящий фильм в title.basics.tsv не найден")


def main() -> None:
    load_env()
    imdb_row = find_first_movie()
    tconst = imdb_row["tconst"]

    find_data = tmdb_get(f"/find/{tconst}", {"external_source": "imdb_id", "language": "ru-RU"})
    results = find_data.get("movie_results", [])
    if not results:
        raise RuntimeError(f"Фильм {tconst} не найден в TMDB")

    tmdb_id = results[0]["id"]
    details = tmdb_get(f"/movie/{tmdb_id}", {"language": "ru-RU"})

    enriched = {
        "id": tconst,
        "title": details.get("title") or imdb_row["primaryTitle"],
        "original_title": imdb_row["originalTitle"],
        "year": int(imdb_row["startYear"]),
        "minutes": int(imdb_row["runtimeMinutes"]),
        "genres": [] if imdb_row["genres"] == r"\N" else imdb_row["genres"].split(","),
        "countries": [c["name"] for c in details.get("production_countries", [])],
        "description": details.get("overview", ""),
    }
    print(json.dumps(enriched, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Киноманьяк: подбор фильмов по каталогу (учебный проект)

Программа подбирает от одного до трёх фильмов из локального каталога `films.json` по запросу пользователя. Модель отвечает только за выбор `film_id` и короткое объяснение (`why`), а название, год, длительность и описание берутся строго из каталога. Код сам проверяет, что в ответе нет выдуманных или повторяющихся фильмов, и записывает каждый вызов в журнал `calls.jsonl`.


## Вопросы для README

Короткие письменные ответы, в каждом — ссылка на файл и строки своего кода.

1. Почему программа сверяет `id`, хотя название понятнее? Сколько расхождений с каталогом дал опыт с памятью модели?
> Для языка понятнее цифры, АЙДИ это как уникальный ключ, который никак не может повторится, поэтому сравнивая его мы гарантируем точное попадание, в отличие от названий.
>9 из 10 расхождений, почти 100%. Модель путала все года, длительность. 
2. Что произойдёт, если модель вернёт `film_id`, которого нет в каталоге? Какие строки кода это ловят?
> в kino.py Мы проверим АЙДИ с нашей базой,  добавим отклоненный АЙДИ в rejected_id, чтобы потмо вывести calls.jsonl. Пользователю выведем "Ответ модели отклонён: обнаружены фильмы вне каталога."
```python 
fid = str(item.film_id)
if fid not in by_id:
    has_foreign = True
    rejected_ids.append(fid)
    continue
if has_foreign or not valid_picks:
    outcome = "invalid"
    ids = []
    print("Ответ модели отклонён: обнаружены фильмы вне каталога.")
```
Как раз во время тестового прогона, на 11 запросе из  queries.json, алгоритм поймал от модели выдуманный id и не пропустив его до клиента отправили в журнал со статусом invalid.

3. Чем «модель ответила без ошибки» отличается от «модель подобрала верный фильм»? Как это видно в `check.py`?

   ```json
   [
     {
       "query": "фильм как Интерстеллар, но не про космос",
       "expect": {"must": ["f-014"], "forbid": ["f-001"]}
     },
     {
       "query": "мультфильм 2031 года",
       "expect": {"no_match": true}
     },
     {
       "query": "қазақша комедия"
     }
   ]
   ```
>"Модель подобрала верный фильм" отличается от "модель ответила без ошибки"отличается качеством контента и его наличие. Если модель ответлиа без ошибок, значит она завернула в схему, то что мы просили, но его содержимое может не соответствовать нашим ожиданиям, мало того может не соответствовать даже ID который мы отдали 
>"Модель подобрала верный фильм"
 ```python
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
  ```
>"Модель ответила без ошибки"
 ```python
      try:
      response = client.chat.completions.parse(**kwargs)
      usage = getattr(response, "usage", None)
      if usage is not None:
          prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
          completion_tokens = getattr(usage, "completion_tokens", 0) or 0

      choice = response.choices[0]
      finish_reason = getattr(choice, "finish_reason", None)
      message = choice.message

      if getattr(message, "refusal", None):
          outcome = "refusal"
          print(f"Отказ модели: {message.refusal}")
      elif finish_reason == "length":
          outcome = "length"
          print("Ответ модели обрезан по лимиту токенов.")

          итд
  ```
4. Какое решение без ML записано в пяти вопросах лекции 1 и чего ему не хватает на запросах из `queries.json`?
5. Сколько стоит один запрос и что сильнее всего влияет на цену?
> Один запрос стоит Исходящие токены* Исходящий тариф + ВходящиеТокены*Входящий тариф, сильнее всего влияет наш большой каталог фильмов с описанием
>$0.001722 в среднем за запрос


## Датасет

Тестовый датасет `films.json` состоит из **51 фильма**:
- Базовые записи отбираются из `title.basics.tsv` и обогащаются данными из `title.akas.tsv` и TMDB API (название, страны, языки, описание).
- Казахстанские фильмы добираются отдельно через TMDB API (`/discover/movie` с `with_origin_country=KZ`).

Генерация датасета с выбором количества фильмов:

```powershell
uv run python generate_films.py --count 40 --kz-count 12
```


## Acknowledgements & Data Source
Movie metadata is provided by IMDb non-commercial datasets.
> Information courtesy of [IMDb](https://www.imdb.com). Used with permission.

Some information from IMDb will riched by tmdb database.
>This product uses the TMDB API but is not endorsed or certified by TMDB.

## Сравнение моделей (Rare)

Прогон 30 запросов из `queries.json` по 3 раза на каждой модели (`uv run check.py --model <модель> --runs 3`):

| Модель | Доля пройденных | Чужие / нарушающие `id` (за 3 прогона) | Среднее время прогона | Суммарная стоимость (3 прогона) |
| --- | --- | --- | --- | --- |
| `gpt-4o-mini` | 77.8% | 0 | 76.2 сек | \$0.043417 |
| `gpt-4o` | 83.3% | 2 | 51.0 сек | \$0.789762 |

## Сравнение моделей (Rare)

Прогон 30 запросов из `queries.json` по 3 раза на каждой модели (`uv run check.py --model <модель> --runs 3`):

| Модель | Доля пройденных | Чужие / нарушающие `id` (за 3 прогона) | Среднее время прогона | Суммарная стоимость (3 прогона) |
| --- | --- | --- | --- | --- |
| `gpt-4o-mini` | 77.8% | 0 | 76.2 сек | \$0.043417 |
| `gpt-4o` | 83.3% | 2 | 51.0 сек | \$0.789762 |

# findex

[![CI](https://github.com/daideneg28-prog/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/daideneg28-prog/findex/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/daideneg28-prog/findex)](https://github.com/daideneg28-prog/findex/releases)

Невелика пошукова система, яку я будую протягом семестру в курсі
«Python програмування» ([опис курсу](https://github.com/rmalkevy/Programming-Practice-Projects/blob/main/courses/python/README.md)).
Кожна лабораторна додає одну можливість мови й один крок проєкту; кожна позначена git-тегом `lab-0N`.

## Корпус

**Project Gutenberg**: 150 англомовних книг (~86 MB тексту, ~15 млн токенів), одна книга = один документ.
Корпус не зберігається в git (`data/` у `.gitignore`). Щоб отримати його:

```bash
uv run python scripts/download_corpus.py --count 150
```

Скрипт бере список англомовних текстів з офіційного каталогу `pg_catalog.csv`, завантажує UTF-8 версії
в `data/gutenberg/<id>.txt` і вирізає ліцензійний заголовок і підвал Gutenberg (з паузою 1 с між запитами).

## Встановлення

Як інструмент командного рядка, з wheel у [GitHub Release](https://github.com/daideneg28-prog/findex/releases):

```bash
uv tool install https://github.com/daideneg28-prog/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl
findex --help
```

Для розробки:

```bash
uv sync
uv run pytest            # тести (+ --cov для покриття)
uv run ruff check .      # лінтер
uv run pyright           # типи, strict
uv tool install .        # `findex` у PATH з робочої копії
```

---

## Лаб 1: ітератори, генератори та корпус

**Що зроблено**

| Модуль | Що робить |
|---|---|
| `src/findex/corpus.py` | `iter_documents(root)`: генератор `Document(doc_id, path, text)`. Обходить теку ліниво через `rglob`, підтримує `.txt` і `.jsonl`. Файл з битим кодуванням не валить програму: пише попередження в лог і читає з `errors="replace"`. |
| `src/findex/tokenize.py` | `tokenize(text)`: генератор токенів через `re.finditer` після NFC і `casefold()`. |
| `src/findex/stats.py` | Конвеєр `документи → токени → Counter` за один прохід. CLI `python -m findex.stats data/` друкує кількість документів і токенів, розмір словника, топ-50 термів, час і пікову пам'ять (`tracemalloc`). `--limit N` реалізовано через `itertools.islice`. `--mode eager` вмикає навмисно «жадібну» версію для порівняння. |
| `tests/test_tokenize.py` | 10 перевірок токенізатора: регістр, кирилиця, комбіновані діакритики, пунктуація, апострофи, дефіси, цифри, порожній рядок. |

Те, що обидві функції є генераторами, перевіряє `inspect.isgeneratorfunction(tokenize)` у тестах.
`next(iter_documents(Path("data")))` повертає першу книгу одразу, не читаючи решту корпусу.

### Правила токенізації

1. **Нормалізація**: спочатку Unicode NFC, потім `str.casefold()`. NFC об'єднує `e` + `U+0301` в один символ `é`, тож
   `café` записане двома способами дає один токен. `casefold()` агресивніший за `lower()`: `"Straße"` → `"strasse"`, тому збігається з `"STRASSE"`.
2. **Токен** — послідовність літер і цифр будь-якої мови (`[^\W_]`, `_` не входить). Комбіновані знаки, що лишилися після NFC
   (наголос над кириличною `а́`, для якої немає готового символу), лишаються всередині слова й не розривають його.
3. **Апострофи** між літерами зберігають слово цілим: `don't`, `п'ять`. Варіанти `’` (U+2019) і `ʼ` (U+02BC) зводяться до `'`,
   тож усі написання `п'ять` дають один токен. Апострофи на початку чи в кінці слова (лапки) відкидаються.
4. **Дефіси** розділяють слова: `state-of-the-art` → `state`, `of`, `the`, `art`, щоб запит `art` знаходив такий документ.
5. **Цифри** — окремі токени: `1984` → `1984`; `3.14` → `3`, `14`; змішані послідовності на кшталт `covid19` лишаються одним токеном.

### Результати на всьому корпусі

```
$ uv run python -m findex.stats data/
documents:   150
tokens:      15,025,665
vocabulary:  200,251
elapsed:     45.01 s
peak memory: 85.8 MiB
top: the 782,327 · and 499,954 · of 430,196 · to 333,918 · a 274,585 · ...
```

### Eager проти lazy

Обидві версії рахують те саме на однаковому зрізі корпусу (`--limit N --mode eager|lazy`).
Python 3.14.5, Windows 11, 16 GB RAM; пам'ять міряє `tracemalloc` (пік), час — `time.perf_counter()`.

| Версія | Документи | Токени | Пікова пам'ять | Час |
|---|---:|---:|---:|---:|
| eager (списки) | 20 | 3 241 152 | 219.2 MiB | 9.47 s |
| lazy (генератори) | 20 | 3 241 152 | 85.8 MiB | 9.28 s |
| eager (списки) | 75 | 9 344 491 | 630.6 MiB | 28.31 s |
| lazy (генератори) | 75 | 9 344 491 | 85.8 MiB | 27.55 s |
| eager (списки) | 150 | 15 025 665 | 1015.6 MiB | 45.85 s |
| lazy (генератори) | 150 | 15 025 665 | 85.8 MiB | 45.01 s |

**Чому так.** Пам'ять eager-версії росте лінійно з розміром корпусу: близько 65 байтів на кожен токен.
Вона одночасно тримає три речі: список усіх текстів (~86 MB рядків), список списків токенів і плаский список `flat`.
Кожен з 15 млн токенів — окремий об'єкт `str` (~50–60 байтів з заголовком) плюс 8-байтовий вказівник у кожному списку, де він лежить.
У lazy-версії пам'ять **однакова для 20, 75 і 150 книг**, тобто не залежить від розміру корпусу.
У кожен момент у пам'яті лише одна книга, її нормалізовані копії і `Counter`.

Пам'ять lazy-версії **не нульова** з двох причин:
1. `Counter` — це відповідь, і він мусить рости: 200 тис. унікальних слів разом з рядками й хеш-таблицею займають десятки MiB.
2. Найбільший документ (`100.txt`, повний Шекспір, 5.5 MB) у моменті існує в кількох копіях: прочитаний текст, результат
   `normalize("NFC")` і результат `casefold()`. Через не-ASCII символи (`’`) Python зберігає його по 2 байти на символ, тож пік
   припадає саме на цю книгу.

Час майже однаковий, бо роботу виконує той самий токенізатор. Eager трохи повільніший через виділення пам'яті під мільйони
об'єктів і розширення списків. Генератори дають виграш у пам'яті, а не в швидкості.

---

## Лаб 2: інвертований індекс, словники, хешування та пам'ять

**Що зроблено**

| Модуль | Що робить |
|---|---|
| `src/findex/index.py` | `build_index(docs, representation, positions)`: один потоковий прохід по корпусу. Для кожного документа рахує власний `Counter` і після документа «скидає» його в глобальний `defaultdict(list)`. `doc_id` призначаються по порядку потоку, тому кожен список постингів **уже відсортований** за `doc_id`, без окремого сортування. Також будує `doc_lengths` (знадобиться для BM25 у Лаб 3) і `doc_meta` (назва книги з `titles.json`). |
| | `Posting` і `DocMeta` мають `@dataclass(frozen=True, slots=True)`: незмінні, хешовані, без `__dict__`. `PlainPosting` — той самий клас без `slots`, лише для порівняння пам'яті. |
| `src/findex/search.py` | Булевий пошук: `a b` (неявне AND), `OR`, `NOT`, обчислення зліва направо. AND/OR/NOT реалізовані **злиттям двома вказівниками** по відсортованих списках (O(n+m), без хешування). `--engine set` вмикає ту саму логіку через `set`. |
| `src/findex/store.py` | `save`/`load` у двох форматах: `.pkl` (pickle) і `.json` (стовпчиковий: `term → [[doc_ids], [tfs]]`). |
| `scripts/bench_lab02.py` | Відтворює всі три таблиці нижче. |
| `tests/test_index.py` | 20 тестів: сортування й tf, хешованість, позиції, злиття = множини, парсер запиту, булеві запити на обох рушіях, round-trip збереження для всіх представлень і форматів, регресія CLI. |

```
$ uv run python -m findex.index data/ --out data/index.pkl
documents:    150
terms:        200,251
postings:     1,086,342
build:        48.96 s
index in RAM: 98.4 MiB (peak 132.8 MiB)
saved:        data\index.pkl (18.4 MiB, 3.93 s)

$ uv run python -m findex.search data/index.pkl "whale ship NOT shakespeare"
10 documents match 'whale ship NOT shakespeare'  (engine=merge)
  [   1] The King James Version of the Bible  (10)
  [   4] The Tragedy of Pudd'nhead Wilson  (102)
  ...
load: 10.05 s · search: 0.203 ms · peak memory: 255.6 MiB
```

(Завантаження в CLI повільніше, ніж у таблиці, бо під час нього працює `tracemalloc`. Він перехоплює кожне виділення пам'яті,
а pickle створює мільйон об'єктів.)

> ⚠️ **pickle небезпечний.** `pickle.load` виконує довільний код: pickle-файл — це програма для unpickler-а, і через
> `__reduce__` вона може викликати будь-яку функцію, наприклад `os.system`. Тому `load()` можна застосовувати **лише до
> індексів, які ви зібрали самі**. Ніколи не завантажуйте pickle з інтернету чи з мережі. JSON-формат безпечний: він може
> створити тільки словники, списки, рядки й числа.

**Баг, на якому я дещо зрозумів.** Перша версія `python -m findex.index` створювала pickle, який `findex.search` не міг
прочитати: `AttributeError: module '__main__' has no attribute 'Posting'`. Під `-m` файл виконується як модуль `__main__`,
тому pickle записав класи як `__main__.Posting`. Pickle зберігає не сам клас, а **шлях імпорту** до нього. Виправлення:
блок `if __name__ == "__main__"` імпортує справжній `findex.index` і викликає `main()` звідти. Тест `test_cli_pickle_is_loadable`
стежить, щоб баг не повернувся.

### Пам'ять: три представлення постингів

150 документів, 200 251 терм, 1 086 342 постинги. Python 3.14.5.

| Представлення постингів | Індекс у RAM | Пік (побудова) | Побудова | Pickle-файл | Завантаження pickle |
|---|---:|---:|---:|---:|---:|
| `list[Posting]`, звичайний `@dataclass` | 139.8 MiB | 157.4 MiB | 47.8 s | 24.6 MiB | 1.02 s |
| `list[Posting]`, `slots=True` | 98.4 MiB | 132.8 MiB | 47.7 s | 18.4 MiB | 4.30 s |
| пари `array('I')` (doc_ids, tfs) | **72.2 MiB** | 123.5 MiB | 45.1 s | 17.0 MiB | **0.34 s** |
| `list[Posting]`, `slots=True` + позиції | 720.1 MiB | 727.9 MiB | 68.8 s | 78.1 MiB | 5.56 s |

**Куди пішли байти** (розміри з `sys.getsizeof` на цій машині):

- **`slots=True`**: об'єкт `Posting` займає 56 B (заголовок і три слоти по 8 B), плюс 8 B вказівник у списку, разом **~64 B на постинг**.
  1.09 млн × 64 B ≈ 66 MiB. Решту ~32 MiB становлять 200 тис. рядків-термів, хеш-таблиця словника і 200 тис. об'єктів `list` (56 B порожній).
- **Звичайний `@dataclass`**: +41 MiB, тобто **~38 B зайвих на постинг**. Це менше, ніж «~100 B на `__dict__`» з теорії, бо з Python 3.13
  атрибути звичайних об'єктів зберігаються «inline», а справжній `__dict__` (296 B) створюється лише при зверненні до нього.
  `slots` усе одно виграє 30%.
- **`array('I')`**: кожен постинг займає рівно **8 B** (два 4-байтові `uint32`), без об'єктів. Самі дані — лише 8.3 MiB.
  Але виграш лише ~26%, а не 8×, бо в нашому корпусі **постингів на терм мало** (у середньому 5.4, більшість термів трапляється
  в одній книзі). Кожен терм несе фіксовані витрати: два порожні `array` по 80 B, кортеж, рядок-терм і слот у словнику. На 200 тис.
  термів це ~60 MiB, які однакові для всіх варіантів. `array` перемагає на довгих списках; що більше документів, то більший розрив.
- **Позиції** збільшують індекс у **7.3 рази**. 15 млн позицій — це `int` до 900 000, а кешуються лише малі (−5…256), тож кожна
  позиція займає 28 B на об'єкт `int` і 8 B на вказівник у кортежі. Вмикаю позиції тільки прапорцем `--positions`.

**Несподіваний результат: pickle зі `slots` завантажується вчетверо повільніше.** Для `frozen=True, slots=True` dataclass
генерує Python-функцію `__setstate__` (`_dataclass_setstate`), і unpickler викликає її для кожного з мільйона об'єктів.
Звичайний клас заповнюється на рівні C. `array` серіалізується як один шматок байтів, тому завантажується найшвидше (0.34 s).

### Формати файлу (індекс зі `slots`)

| Формат | Розмір | Збереження | Завантаження |
|---|---:|---:|---:|
| pickle | 18.4 MiB | 3.44 s | 4.26 s |
| JSON (стовпчиковий) | **8.9 MiB** | **2.36 s** | **2.29 s** |

JSON тут виграв за всіма показниками, бо формат **стовпчиковий**: на терм два списки чисел, а не мільйон окремих об'єктів з
іменем класу й полями. Pickle зберігає граф об'єктів, і на кожен `Posting` припадають опкоди виклику конструктора.
Звичайне правило «pickle швидший за JSON» стосується однакової структури даних, а не однакової інформації.

### Злиття двома вказівниками проти `set`

| Запит | Розміри постингів | Рушій | AND | OR |
|---|---|---|---:|---:|
| два найчастіші: `of` + `in` | 150 + 150 | merge | 17.86 µs | 18.01 µs |
| два найчастіші: `of` + `in` | 150 + 150 | set | **7.81 µs** | **7.07 µs** |
| два найрідші: `diskpack` + `achsa` | 1 + 1 | merge | **0.27 µs** | **0.46 µs** |
| два найрідші: `diskpack` + `achsa` | 1 + 1 | set | 0.44 µs | 0.48 µs |

Асимптотично обидва варіанти O(n+m). Але злиття — цикл **на Python**, кожен крок якого інтерпретується, а `set(a) & set(b)`
хешує й перетинає **на C**. На довгих списках C перемагає вдвічі, попри зайве `sorted()`. На коротких списках виграє злиття,
бо `set` платить фіксовану ціну за створення двох множин. Злиття лишається основним рушієм з причин, які важать більше
за швидкість у CPython: воно потокове, не потребує пам'яті під хеш-таблиці, працює зі стиснутими списками й розширюється до
фразового пошуку в Лаб 3. Перевагу C-коду для довгих списків повернемо в Лаб 8 через NumPy.

> Починаючи з Лаб 3 CLI пошуку змінився (ранжування замість булевого списку), а операції злиття переїхали в `findex/merge.py`.
> Стан коду саме Лаб 2 доступний за тегом `lab-02`.

---

## Лаб 3: модель об'єктів, ранжування, мова запитів

**Що зроблено**

| Модуль | Що робить |
|---|---|
| `src/findex/index.py` | `Index` успадковує `collections.abc.Mapping[str, list[Posting]]`. Реалізовані `__getitem__` (`KeyError` для невідомого терма), `__iter__`, `__len__` (розмір словника), швидкий `__contains__` і `__repr__`. `keys/items/values/get` дає `Mapping`. Є `num_docs`, `avg_doc_length` (`@cached_property`), `doc_length()`, `df()`. `open_index(path)` — контекстний менеджер через `@contextmanager`: індекс закривається навіть при винятку. |
| `src/findex/rank.py` | Протокол `Scorer` (`typing.Protocol`, структурна типізація без наслідування) і дві реалізації: `TfIdf` і `BM25(k1=1.5, b=0.75)`. `search()` накопичує бали в словнику й бере top-k через `heapq.nlargest`. Результати — `SearchResult` з `order=True`. Співставлення запиту кешується через `@lru_cache(maxsize=256)`. |
| `src/findex/query.py` | Рекурсивний спуск: `or_expr → and_expr ("OR" and_expr)*`, `and_expr → not_expr (["AND"] not_expr)*`, `not_expr → "NOT" not_expr \| atom`, `atom → слово \| "фраза" \| ( or_expr )`. Вузли `Term`, `Phrase`, `And`, `Or`, `Not` — frozen dataclass-и з перевантаженими `&`, `\|`, `~` і методом `evaluate(index)`. Фрази шукаються злиттям двома вказівниками по **позиціях**. |
| `src/findex/timing.py` | Декоратор `@timed` з `functools.wraps`; стоїть на `build_index`, `load`, `search`, `match`. |
| `src/findex/snippets.py` | Вікно ±80 символів навколо місця, де зібралося найбільше **різних** термів запиту, з підсвіткою `«терм»`. |
| `scripts/eval_lab03.py` | Три перевірки адекватності ранжування і precision@5 нижче. |
| `tests/test_query.py`, `tests/test_rank.py` | Тести парсера **як дерева** (`parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))`), обчислення запитів і фраз, усі три перевірки на мініатюрному корпусі, кеш, `@timed`, сніпети. Усього в проєкті 76 тестів. |

Індекс тепер за замовчуванням будується **з позиціями** (`--no-positions` вимикає). Без позицій фраза
деградує до AND з попередженням у лозі.

```
$ uv run python -m findex.search data/index.pkl 'tarzan "the apes" NOT jane' '"white whale" ahab' \
      '(huck OR huckleberry) "tom sawyer"' 'tarzan "the apes" NOT jane' -k 3
  [log] load took 3580.39 ms
  [log] match took 27.73 ms        ← перший запит: парсинг + обчислення фрази
  [log] search took 27.94 ms
  ...
  [log] match took 0.00 ms         ← той самий запит удруге: результат з lru_cache
  [log] search took 0.12 ms
Index(150 docs, 200,251 terms, representation='slots', positions=True)

=== 'tarzan "the apes" NOT jane' (bm25)
  1 documents match; top 1:
   1.  12.346  Jungle Tales of Tarzan
      …«the» shade of «the» tropical forest, presented, unquestionably, a most alluring picture of young,
      feminine loveliness. Or at least so thought «Tarzan» of «the» «Apes»…

=== '"white whale" ahab' (bm25)
   1.  13.855  Moby-Dick; or, The Whale
      …queer about that, eh? A «white» «whale»—did ye mark that, man? Look ye—there's something special in
      the wind. Stand by for it, Flask. «Ahab» has that that's bloody…

=== '(huck OR huckleberry) "tom sawyer"' (bm25)
  5 documents match; top 3:
   1.  24.616  The Adventures of Tom Sawyer, Complete
   2.  23.309  Adventures of Huckleberry Finn
   3.  17.017  Tom Sawyer, Detective

query cache: 5 hits, 3 misses, 3 cached
```

(У PowerShell 5.1 лапки всередині аргументу треба писати як `\"`, інакше оболонка їх прибирає.)

### Формули

- **TF-IDF**: `(1 + ln tf) · ln(N / df)`.
- **BM25**: `idf · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, де `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`.

### Три перевірки адекватності на корпусі

**1. Рідкісний терм важить більше за частий.** Запит `the OR ahab` (BM25):

| Бал | Документ |
|---:|---|
| 8.736 | Moby-Dick; or, The Whale |
| 7.934 | The King James Version of the Bible |
| 7.907 | The Bible, King James Version, Complete |
| 5.659 | … найкращий документ лише зі словом `the`: Paradise Regained |

`idf(the) = 0.0033` (слово є в усіх 150 книгах), `idf(ahab) = 3.51`. Обидві Біблії тут правомірно: там є цар Ахав.

**2. BM25 насичується за частотою терма.** Внесок `whale` для документа середньої довжини:

| tf | BM25 | TF-IDF |
|---:|---:|---:|
| 1 | 1.931 | 1.920 |
| 2 | 2.748 | 3.250 |
| 5 | 3.681 | 5.009 |
| 10 | 4.151 | 6.340 |
| 19 | 4.419 | 7.572 |
| **20** | **4.435** (+0.016) | 7.670 (+0.098) |
| 100 | 4.691 | 10.760 |

Двадцяте входження додає до BM25 лише 0.016, а межа при `tf → ∞` дорівнює `idf·(k1+1) = 4.759`. TF-IDF росте без обмежень,
тому в «Мобі Діку» (`tf(whale) = 1116`) TF-IDF дав би слову whale величезну вагу.

**3. Нормалізація довжини.** Терм `rich'd` трапляється рівно один раз у двох документах:

| Документ | Довжина | BM25 | TF-IDF |
|---|---:|---:|---:|
| The United States Constitution | 4 727 токенів | **7.179** | 4.317 |
| The Complete Works of William Shakespeare | 968 609 токенів | 0.837 | 4.317 |

Для TF-IDF ці документи однакові. BM25 розуміє, що одне слово в коротенькому тексті значить більше, ніж те саме слово,
загублене серед мільйона інших. При `b = 0` різниця зникає (це перевіряє тест `test_bm25_prefers_short_document`).

### precision@5: TF-IDF проти BM25

10 запитів, релевантні книги розмічені вручну за змістом (список у `scripts/eval_lab03.py`). Терми запиту з'єднані `OR`,
щоб збіг не вимагав усіх слів одночасно.

| Запит | Релевантних | TF-IDF P@5 | BM25 P@5 |
|---|---:|---:|---:|
| `tarzan apes jungle` | 6 | 1.0 | 1.0 |
| `barsoom martian princess` | 4 | 0.8 | 0.8 |
| `tom sawyer huck` | 4 | 0.8 | 0.8 |
| `anne gilbert avonlea` | 3 | 0.6 | 0.6 |
| `internet email network` | 7 | 0.8 | 0.8 |
| `population gdp exports` | 4 | 0.8 | 0.8 |
| `dorothy oz scarecrow` | 2 | 0.4 | 0.4 |
| `inaugural address fellow citizens` | 4 | 0.0 | **0.8** |
| `sara crewe minchin` | 2 | 0.4 | 0.4 |
| `paradise satan eden` | 3 | 0.4 | **0.6** |
| **середнє** | | **0.60** | **0.70** |

Для запитів, де релевантних книг менше п'яти, максимум P@5 обмежений: 2 релевантні дають не більше 0.4, 3 — не більше 0.6.
На них обидва скорери досягли стелі. Різниця лише у двох запитах, і обидва раз пояснюються **нормалізацією довжини**.
Інавгураційні промови займають кілька сторінок, тож за TF-IDF їх витісняють товсті томи, де «fellow citizens» трапляється
випадково (переписи, «Федераліст»). BM25 піднімає короткі документи, в яких ці слова щільні. Вибірка мала й шумна,
тож висновок обережний: на цьому корпусі BM25 не гірший ніде і кращий там, де довжина документів дуже різна.

---

## Лаб 4: типи, тести, пакування і CLI

**Що зроблено**

- **Типи: `pyright --strict` дає 0 помилок** на `src/` і `tests/`, без жодного `# type: ignore`.
  - Найбільше роботи було з постингами. Раніше тип був «`list` або пара `array`», а розгалуження йшло за рядком
    `representation`, і pyright не міг вивести тип. Тепер є `type PostingList = list[PostingLike] | ArrayPostings`
    (аліаси PEP 695) і звуження через `isinstance(entry, tuple)`, яке перевіряльник розуміє.
  - `PostingLike` — `typing.Protocol`, який задовольняють і `Posting`, і `PlainPosting`.
  - `Scorer` — теж `Protocol`.
  - Генератори типізовані як `Iterator[...]`, контекстний менеджер — як `Generator[Index]`.
  - `@timed` має сигнатуру `def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]` (PEP 695 + `ParamSpec`): після
    декорування pyright і далі бачить справжні параметри функції.
- **CLI `findex`** (`typer` + `rich`, `[project.scripts] findex = "findex.cli:app"`):
  - `findex index <corpus> --out ix.pkl [--no-positions] [--limit N] [--repr slots|plain|array]` з прогрес-баром `rich`;
  - `findex search <index> "<query>" [-k 10] [--scorer bm25|tfidf] [--json] [--no-snippets]`: таблиця з підсвіченими
    збігами у фрагментах тексту, або JSON lines для `| jq`;
  - `findex stats <index> [--top 20]`;
  - `-v` / `-vv` перемикають рівень логування.
  - **Результати йдуть у stdout, а все діагностичне (логи, прогрес, помилки) у stderr**, тому `--json` можна безпечно
    передавати далі пайпом.
  - Коди виходу: `0` — успіх (навіть без результатів), `1` — синтаксична помилка запиту, `2` — немає файлу або неприпустимі
    параметри (наприклад, `--repr array` разом з позиціями).
- **Логування** налаштовується один раз у `cli.py` через `RichHandler` у stderr; модулі тільки викликають `logging.getLogger(__name__)`.
- **CI** (`.github/workflows/ci.yml`) на кожен push і PR:
  1. ruff (lint + format);
  2. pyright strict;
  3. pytest з coverage;
  4. збирає wheel, ставить його через `uv tool install` у чисте середовище й запускає `findex --help`.
- **Реліз** [`v0.4.0`](https://github.com/daideneg28-prog/findex/releases/tag/v0.4.0) з wheel і sdist.

```
$ findex -v search data/index.pkl 'barsoom "dejah thoris"' -k 3
INFO     load took 3611.77 ms
INFO     search took 1.35 ms
INFO     4 documents match 'barsoom "dejah thoris"'
                   4 documents match 'barsoom "dejah thoris"' (bm25)
┌───┬────────┬─────────────────────┬──────────────────────────────────────────────────────────────┐
│ # │  score │ title               │ snippet                                                      │
├───┼────────┼─────────────────────┼──────────────────────────────────────────────────────────────┤
│ 1 │ 26.109 │ A princess of Mars  │ …rks of Barsoom. CHAPTER XI WITH DEJAH THORIS As we reached… │
│ 2 │ 25.969 │ The gods of Mars    │ …who had won the ever beautiful Dejah Thoris, Princess of H… │
│ 3 │ 25.966 │ The warlord of Mars │ …as Dejah Thoris looked up full into my eyes I made the sig… │
└───┴────────┴─────────────────────┴──────────────────────────────────────────────────────────────┘

$ findex search data/index.pkl 'white whale' --json -k 1
{"rank": 1, "score": 5.1277, "title": "Moby-Dick; or, The Whale", "snippet": "…he caught once a whale in Spitzbergen that was white all over.…"}

$ findex search data/index.pkl '(broken'; echo $?
query error: missing ')'
1
```

### Тести

**99 тестів** (`uv run pytest`): 94 звичайні й 5 властивостей Hypothesis.

| Файл | Що перевіряє |
|---|---|
| `tests/conftest.py` | спільний корпус: 4 файли + `titles.json` у `tmp_path`, індекс з нього, збережений `.pkl` |
| `test_tokenize.py` | регістр/`casefold`, кирилиця, комбіновані діакритики, пунктуація, апострофи, дефіси, цифри |
| `test_corpus_stats.py` | лінивість `iter_documents`, JSONL, бите кодування (лог замість падіння), lazy == eager |
| `test_index.py` | протокол `Mapping`, статистики, хешованість, позиції, злиття, `open_index` закривається при винятку, round-trip pickle/JSON, регресія `__main__` (`slow`) |
| `test_query.py` | парсер **як дерева**, `&`/`\|`/`~`, синтаксичні помилки, обчислення фраз |
| `test_rank.py` | три перевірки адекватності з Лаб 3 як тести, top-k, кеш, `@timed` + `wraps`, сніпети |
| `test_cli.py` | усі підкоманди через `typer.testing.CliRunner`: таблиця, `--json`, коди виходу, логи саме в stderr |
| `test_properties.py` | **5 властивостей Hypothesis**: злиття = операції множин; токенізація ідемпотентна; токени вже нормалізовані (NFC, casefold, без пробілів); індекс точно відповідає документам (df, tf, позиції, довжини); JSON round-trip зберігає постинги |

`@pytest.mark.slow` позначає тест, що запускає підпроцес Python (`pytest -m "not slow"` його пропускає).

> На моєму ноутбуці Windows Smart App Control блокує нативне розширення Hypothesis (`ImportError: An Application Control
> policy has blocked this file`), тому локально `test_properties.py` пропускається з цим поясненням
> (`pytest.importorskip(..., exc_type=ImportError)`). **У CI на Linux властивості виконуються завжди.**
> З тієї ж причини `pytest` і `pyright` запускаю як `python -m pytest` і `python -m pyright`: `.exe`-обгортки теж блокуються.

**Покриття: 90%** (`uv run pytest --cov`, branch coverage). Що не покрито і чому:

| Модуль | Покриття | Пояснення |
|---|---:|---|
| `search.py` | 0% | старий CLI Лаб 3 (`python -m findex.search`), його замінив `findex search`; лишається для відтворення README Лаб 3 |
| `__main__.py` | 0% | дві рядки, викликає той самий `app`, що й тести CLI |
| `index.py` | 87% | `main()` модульного CLI Лаб 2 (`python -m findex.index`) перевіряє лише `slow`-тест у підпроцесі, а coverage не бачить підпроцесів |
| `cli.py` | 90% | гілка, де фрагмент тексту не вдалося прочитати (файл книги видалили після індексації), і рядки, які `rich` друкує лише у справжньому терміналі |

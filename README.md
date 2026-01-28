# Скрипт для анализа турниров на ПокерОк

Консольный анализатор результатов турниров (MTT) для PokerOK: ведение базы сыгранных турниров, расчёт ключевых метрик (ROI, ITM, ABI, профит), отчёты по фильтрам, импорт/экспорт CSV и упрощённая симуляция дисперсии (Monte-Carlo).

---

## Возможности

### ✅ Учёт турниров PokerOK

* Добавление турнира в базу (`add`) — через аргументы CLI или интерактивно.
* Локальное хранение данных в `tournaments.json` рядом со скриптом.
* Поддержка типов турниров: `MTT`, `Bounty`, `Turbo`, `Hyper`, `Satellite`, `Other`.

### ✅ Аналитика и метрики

* **Total Cost / Total Cash / Profit**
* **ROI%** (прибыль / затраты)
* **ITM%** (доля турниров с выигрышем)
* **ABI** (Average Buy-In: средняя стоимость турнира с учётом рейка и ре-энтри)
* **Avg Profit** на турнир
* **Best / Worst** результат
* **Max Drawdown** (максимальная просадка по кривой профита)

### ✅ Отчёты и фильтры

* Отчёт по периоду, диапазону стоимости, типу турнира и валюте (`report`).
* Топ-5 лучших и худших турниров в выбранной выборке.

### ✅ Импорт / экспорт

* Импорт результатов из CSV (`import`) с логом ошибок `import_errors.csv`.
* Экспорт базы или отфильтрованной выборки в CSV (`export`).

### ✅ Симуляция дисперсии (Variance Simulation)

* Monte-Carlo симуляция результата на дистанции N турниров (`simulate`).
* Перцентили результата (P5/P50/P95) и вероятность оказаться в минусе.

> ⚠️ Симуляция использует **упрощённую нормальную модель** и не учитывает реальную структуру выплат MTT (ICM, баунти, призовые зоны и т.п.). Это инструмент для интуитивной оценки «размаха» результатов.

---

## Установка

### Требования

* Python **3.10+** (подойдёт и 3.9+, но рекомендуется 3.10+)
* Внешние библиотеки **не нужны** (только стандартная библиотека Python)

### Клонирование

```bash
git clone https://github.com/PokerOk-GGG/Tournament-Profit-Analyzer.git
cd Tournament-Profit-Analyzer
```

---

## Быстрый старт

### 1) Добавить турнир (интерактивно)

```bash
python tournament.py add
```

### 2) Посмотреть общую статистику

```bash
python tournament.py stats
```

### 3) Отчёт за период

```bash
python tournament.py report --from 2025-12-01 --to 2025-12-31
```

### 4) Симуляция дисперсии на дистанции

```bash
python tournament.py simulate --n 1000 --abi 5.5 --roi 0.20 --sigma 1.6
```

### 5) Экспорт в CSV

```bash
python tournament.py export --out export.csv
```

---

## Команды и примеры

## `add` — добавить один турнир

Можно передать параметры сразу:

```bash
python tournament.py add --date 2025-12-16 --buyin 5 --rake 0.5 --cash 0 --type MTT --field 1200
```

Поддерживаемые параметры:

* `--date` — дата `YYYY-MM-DD`
* `--type` — тип турнира (`MTT`, `Bounty`, `Turbo`, `Hyper`, `Satellite`, `Other`)
* `--buyin` — бай-ин (без рейка)
* `--rake` — комиссия
* `--cash` — выигрыш (0 если без ITM)
* `--reentries` — количество ре-энтри (0 = без ре-энтри)
* `--currency` — валюта (по умолчанию `USD`)
* `--field` — размер поля (опционально)
* `--place` — место (опционально)
* `--notes` — заметки (опционально)

---

## `stats` — статистика по всей базе

```bash
python tournament.py stats
```

Фильтр по валюте:

```bash
python tournament.py stats --currency USD
```

---

## `report` — отчёт по фильтрам

Примеры:

За период:

```bash
python tournament.py report --from 2025-11-01 --to 2025-12-16
```

По диапазону стоимости турнира (**фильтр применяется по `total_cost`**, то есть buyin+rake и ре-энтри учитываются):

```bash
python tournament.py report --min-buyin 1 --max-buyin 10
```

По типу:

```bash
python tournament.py report --type Bounty
```

Комбинированный фильтр:

```bash
python tournament.py report --from 2025-12-01 --to 2025-12-31 --type MTT --min-buyin 2 --max-buyin 15 --currency USD
```

---

## `import` — импорт турниров из CSV

Минимально:

```bash
python tournament.py import my_tournaments.csv
```

С указанием валюты по умолчанию (если в CSV нет `currency`):

```bash
python tournament.py import my_tournaments.csv --currency USD
```

Ошибки импорта сохраняются в файл:

* по умолчанию `import_errors.csv`
* либо указать:

```bash
python tournament.py import my_tournaments.csv --errors my_import_errors.csv
```

---

## `export` — экспорт в CSV

```bash
python tournament.py export --out export.csv
```

Можно экспортировать только выборку:

```bash
python tournament.py export --from 2025-12-01 --to 2025-12-31 --type MTT --out december_mtt.csv
```

---

## `simulate` — симуляция дисперсии

Пример:

```bash
python tournament.py simulate --n 2000 --abi 5.5 --roi 0.25 --sigma 1.6 --trials 5000
```

Параметры:

* `--n` — количество турниров на дистанции
* `--abi` — средняя стоимость турнира (total_cost)
* `--roi` — ожидаемый ROI **долей** (например `0.2` = 20%)
* `--sigma` — «ширина» распределения в ABI-единицах (примерно 1.3–2.5, зависит от поля/структуры)
* `--trials` — число прогонов Monte-Carlo (по умолчанию 5000)
* `--loss-threshold` — вероятность результата хуже заданного значения:

```bash
python tournament.py simulate --n 1000 --abi 5.5 --roi 0.2 --sigma 1.6 --loss-threshold -1000
```

---

## `reset` — очистка базы

⚠️ Удаляет все сохранённые турниры.

```bash
python tournament.py reset
```

Для подтверждения нужно ввести `RESET`.

---

## Формулы (как считает скрипт)

### Стоимость турнира (учёт re-entry)

* `entries = 1 + reentries`
* `total_cost = (buyin + rake) * entries`

### Профит и ROI по турниру

* `profit = cash - total_cost`
* `roi = profit / total_cost`

### Метрики по выборке турниров

* `TotalCost = sum(total_cost)`
* `TotalCash = sum(cash)`
* `Profit = TotalCash - TotalCost`
* `ROI% = Profit / TotalCost * 100`
* `ITM% = count(cash > 0) / N * 100`
* `ABI = TotalCost / N`

### Max Drawdown

Максимальная просадка считается по кривой накопленного профита, отсортированной по `date`.

---

## Формат CSV для импорта

### Минимальные колонки (обязательные)

```csv
date,buyin,rake,cash,type
2025-12-01,5,0.5,0,MTT
2025-12-02,5,0.5,18,MTT
```

### Расширенный формат (рекомендуется)

```csv
date,buyin,rake,cash,type,currency,reentries,field,place,notes
2025-12-01,5,0.5,0,MTT,USD,0,1200,,late reg
2025-12-02,5,0.5,18,MTT,USD,1,1500,45,2 bullets
```

---

## Хранилище данных

По умолчанию скрипт создаёт/использует файл:

* `tournaments.json`

Можно указать другой путь:

```bash
python tournament.py --store my_data.json stats
```

---

## Дисклеймер

Проект не является официальным инструментом PokerOK и не связан с разработчиками рума. Скрипт не использует API PokerOK, не взаимодействует с клиентом и хранит данные локально.

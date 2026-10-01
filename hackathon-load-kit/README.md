# Нагрузка Kilo Code

Скрипт воспроизводит участников, работающих из Kilo Code, против OpenAI-совместимой ручки `/v1/chat/completions`.

Зависимости: Python 3.11+, `httpx`.

```bash
python3 -m venv .venv
.venv/bin/pip install httpx
export LOADTEST\_API\_KEY='...'
.venv/bin/python kilo\_load.py --self-test


```



.venv/Scripts/python kilo\_load.py steps \\

&#x20; --base-url https://alfagen.alfabank.ru/continue-dev/v1 \\

&#x20; --model deepseek-ai/DeepSeek-V4-Flash-0731 \\

&#x20; --api-key-env LOADTEST\_API\_KEY \\

&#x20; --corpus-root . \\

&#x20; --output ./out/fine.jsonl \\

&#x20; --sessions 64 \\

&#x20; --steps 32,40,48,56,64,128,256,300 \\

&#x20; --step-duration-seconds 240 \\

&#x20; --cooldown-seconds 30



Ключ только в env. URL, модель, имя переменной с ключом, каталог кода и CA — аргументы.

## Сценарии

Общие флаги: `--base-url`, `--model`, `--api-key-env`, `--corpus-root`, `--output`, при необходимости `--ca-file`.

Контекст сессии набирается из файлов `--corpus-root` (не синтетический текст). Между сессиями префикс расходится, внутри сессии не переписывается. `max\_tokens` по умолчанию 32000.

### steps — ступени одновременных сессий

По умолчанию геометрическая лестница до 300 (1, 2, 4, …, 300), вход разнесён, на каждой ступени вердикт SLA и номер первой сломанной ступени.

```bash
.venv/bin/python kilo\_load.py steps \\
  --base-url https://alfagen.alfabank.ru/continue-dev/v1 \\
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \\
  --api-key-env LOADTEST\_API\_KEY \\
  --corpus-root /path/to/repo \\
  --sessions 300 \\
  --output ./out/steps.jsonl
```

Узкая лестница: `--steps 1,2,4,8 --sessions 8`. Длительность ступени: `--step-duration-seconds`.

### burst — одновременный старт

N сессий (по умолчанию 300) стартуют почти сразу. Смотреть отказы и коды HTTP.

```bash
.venv/bin/python kilo\_load.py burst \\
  --base-url https://alfagen.alfabank.ru/continue-dev/v1 \\
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \\
  --api-key-env LOADTEST\_API\_KEY \\
  --corpus-root /path/to/repo \\
  --sessions 300 \\
  --output ./out/burst.jsonl
```

### soak — удержание

N сессий держатся заданное время.

```bash
.venv/bin/python kilo\_load.py soak \\
  --base-url https://alfagen.alfabank.ru/continue-dev/v1 \\
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \\
  --api-key-env LOADTEST\_API\_KEY \\
  --corpus-root /path/to/repo \\
  --sessions 300 \\
  --duration-hours <часы> \\
  --output ./out/soak.jsonl
```

## Отчёт

Рядом с JSONL пишется `\*.summary.json`:

* число запросов и ошибок, коды HTTP;
* TTFT входа (p95, порог 120 с) и TTFT продолжения (p95, порог 30 с);
* скорость генерации p5 (порог 50 tok/s);
* доля ошибок (порог 1 %);
* вердикт SLA; для `steps` — `first\_violation\_sessions`.

Пороги меняются флагами `--slo-ttft-entry-seconds`, `--slo-ttft-continuation-seconds`, `--slo-tokens-per-second`, `--slo-error-rate`.

Профиль человека внутри скрипта: всплеск 8 ходов, пауза между всплесками p50 ≈ 134 с.

Шаблоны запроса: `templates/system\_prompt.txt`, `templates/tools.json`.


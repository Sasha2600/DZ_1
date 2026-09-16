# AGENTS.md

Учебный проект (OTUS ДЗ): демо-агент поддержки `Reason → Act → Observe`
с LLM через OpenAI-совместимый API (LM Studio).

## Запуск

- Зависимости: `pip install -r requirements.txt` (openai==2.54.0, python-dotenv==1.2.2).
- Скопировать `.env.example` → `.env` (`.env` в .gitignore). `agent.py` читает его через `load_dotenv()`.
- Запуск: `python agent.py` — интерактивный CLI-цикл запросов (`exit`/`quit` — выход).

## Обязательное требование

- Нужен запущенный LM Studio Local Server на `http://localhost:1234/v1` с загруженными
  моделью/моделями — без него код падает на сетевой ошибке.
- В `.env` вписать точные Model Identifier (LM Studio → Local Server). Одна модель →
  одно и то же имя в `CHEAP_MODEL` и `ADVANCED_MODEL`.
- `LM_STUDIO_API_KEY` для LM Studio — любое непустое значение (сервер ключ не проверяет).

## Архитектура (весь код в одном файле `agent.py`)

- `classify()` — REASON: guardrail + категория/риск/сложность/язык (JSON от CHEAP_MODEL).
- `retrieve_context()` — ACT: поиск по демо-БД `DEMO_KB` (keyword-скоринг, **не** векторный
  поиск — так задумано для демо, не переделывать).
- `generate_answer()` — ACT: роутинг: простые+уверенные → CHEAP_MODEL, остальное → ADVANCED_MODEL.
- `validate_grounding()` — OBSERVE: проверка, что ответ подтверждён контекстом.
- `run_agent()` — оркестрация цикла с защитой от Dead Loop: `MAX_ITERATIONS`, таймауты цикла
  и запроса, лимит токенов диалога, детект повторного состояния `(answer, model)` —
  зацикливание фиксируется только при повторе ответа от той же модели, что разрешает
  разным моделям (CHEAP → ADVANCED) обрабатывать один запрос.
- Модель возвращает JSON, парсится `_chat_json()`; при сбое парсинга пайплайн деградирует
  в безопасную сторону (эскалация, `_parse_error`), а не падает.

## Конвенции

- Весь код, промпты, комментарии и README — на русском; новый код писать на русском.
- Тестов, линтера, CI, pyproject/setup в репо **нет** — не ищите их.
- `venv/` — локальное окружение (Python 3.13), в git не попадает. Версия Python нигде
  не зафиксирована строго (README: 3.12+, докстринг `agent.py`: 3.14) — доверять `venv/`.
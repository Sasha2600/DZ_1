"""
Агент поддержки: Reason -> Act -> Observe.

Подключается к локальной LLM, запущенной в LM Studio, через OpenAI-совместимый
клиент (openai>=1.0). LM Studio по умолчанию поднимает сервер на
http://localhost:1234/v1 и принимает любой api_key (значение не проверяется).

Требования: Python 3.14, пакеты `openai` и `python-dotenv`
(pip install openai python-dotenv).

Конфигурация читается из переменных окружения / файла `.env` (см.`.env.example`)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Конфигурация — читается из .env / переменных окружения
# --------------------------------------------------------------------------- #

load_dotenv()  # подхватывает файл .env из текущей директории, если он есть

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# Замените на реальные идентификаторы моделей, загруженных в LM Studio,
# либо задайте их через .env.
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemma-4-e2b")
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "google/gemma-4-31b-qat")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
TOTAL_CYCLE_TIMEOUT_SECONDS = int(os.getenv("TOTAL_CYCLE_TIMEOUT_SECONDS", "15"))
MAX_DIALOG_TOKENS_APPROX = int(os.getenv("MAX_DIALOG_TOKENS_APPROX", "8000"))

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)


# --------------------------------------------------------------------------- #
# Вспомогательные типы
# --------------------------------------------------------------------------- #

class Outcome(StrEnum):
    ANSWERED = auto()
    ESCALATED = auto()
    CLARIFICATION_NEEDED = auto()


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class RetrievedDoc:
    document: Document
    score: float


@dataclass
class AgentResult:
    outcome: Outcome
    message: str
    sources: list[str] = field(default_factory=list)
    iterations_used: int = 0
    escalation_reason: str | None = None


# --------------------------------------------------------------------------- #
# База знаний (демо: keyword-скоринг вместо векторного поиска)
# --------------------------------------------------------------------------- #

class KnowledgeBase:
    """
    Упрощённая база знаний. В проде это будет векторный поиск
    (embeddings + faiss/pgvector/etc.), здесь — keyword overlap, чтобы
    пример работал без дополнительной инфраструктуры.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        # Упрощенный список стоп-слов
        stopwords = {"и", "в", "на", "для", "как", "что", "с", "по", "из", "не", "но", "или", "то", "так", "у"}
        query_words = [w for w in query.lower().split() if len(w) > 2 and w not in stopwords]
        
        if not query_words:
            query_words = query.lower().split()
            
        scored: list[RetrievedDoc] = []
        for doc in self._documents:
            doc_text_lower = doc.text.lower()
            doc_words = doc_text_lower.split()
            score = 0
            
            for q_word in query_words:
                match_found = False
                for d_word in doc_words:
                    # Проверка на полное совпадение
                    if q_word == d_word:
                        score += 2
                        match_found = True
                        break
                    # Совпадение первых 4 символов
                    elif len(q_word) >= 4 and len(d_word) >= 4 and q_word[:4] == d_word[:4]:
                        score += 1
                        match_found = True
                        break
                    # Обычное подстрочное совпадение
                    elif q_word in d_word:
                        score += 1
                        match_found = True
                        break
            score = score / max(len(query_words), 1)
            if score > 0:
                scored.append(RetrievedDoc(document=doc, score=round(score, 3)))
        
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


DEMO_KB = KnowledgeBase(
    documents=[
        Document(
            doc_id="faq-refund",
            title="Возврат средств",
            text=(
                "Возврат средств оформляется в течение 14 дней с момента покупки. "
                "Для возврата напишите в поддержку номер заказа и причину возврата. "
                "Деньги возвращаются на исходный способ оплаты в течение 5-7 рабочих дней."
            ),
        ),
        Document(
            doc_id="faq-shipping",
            title="Сроки доставки",
            text=(
                "Стандартная доставка занимает 3-5 рабочих дней по России. "
                "Экспресс-доставка — 1-2 дня, доступна для крупных городов."
            ),
        ),
        Document(
            doc_id="faq-warranty",
            title="Гарантия на товары",
            text=(
                "На все товары действует гарантия от производителя сроком 12 месяцев. "
                "В случае неисправности в течение гарантийного срока, "
                "необходимо обратиться в авторизованный сервисный центр с чеком."
            ),
        ),
        Document(
            doc_id="faq-payment",
            title="Способы оплаты",
            text=(
                "Мы принимаем банковские карты (Visa, MasterCard, Мир), "
                "электронные кошельки и оплату через системы быстрых платежей. "
                "Оплата онлайн доступна в личном кабинете."
            ),
        ),
        Document(
            doc_id="faq-change_address",
            title="Изменение адреса",
            text=(
                "Если вы хотите изменить адрес доставки, это можно сделать в течение "
                "30 минут после оформления заказа. После этого изменения "
                "требуют ручного подтверждения оператором."
            ),
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Вызовы модели
# --------------------------------------------------------------------------- #

def _chat_json(model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Вызывает модель и парсит JSON-ответ (с проверкой на соотвествие структуре ответа формату json)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Модель не всегда строго следует формату — не роняем пайплайн,
        # а деградируем в безопасную сторону (эскалация на следующем шаге).
        return {"_parse_error": True, "_raw": raw}


def _chat_text(model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


def _approx_tokens(*texts: str) -> int:
    # Грубая оценка для русского языка: ~4 символа на токен. 
    return sum(len(t) for t in texts) // 4


# --------------------------------------------------------------------------- #
# ЭТАП 1 — REASON: классификация запроса (guardrail + маршрутизация)
# --------------------------------------------------------------------------- #

CLASSIFY_SYSTEM_PROMPT = """\
Ты — классификатор входящих запросов в поддержку.
Твоя задача: проанализировать запрос пользователя и определить его категорию, уровень риска, сложность и язык.

Сначала кратко проанализируй запрос, а затем верни ТОЛЬКО JSON без пояснений и без markdown-разметки в формате:
{
  "category": "faq" | "complaint" | "other",
  "risk": "low" | "high",
  "complexity": "simple" | "complex",
  "language": "ru" | "en" | "other"
}
risk="high" — если запрос содержит жалобу, угрозу, юридическую тему,
упоминание вреда себе или другим, оскорбления.
complexity="complex" — если вопрос требует синтеза нескольких фактов
или неоднозначен.
"""


def classify(query: str) -> dict[str, Any]:
    return _chat_json(CHEAP_MODEL, CLASSIFY_SYSTEM_PROMPT, query)


# --------------------------------------------------------------------------- #
# ЭТАП 2 — ACT: retrieval
# --------------------------------------------------------------------------- #

def retrieve_context(query: str) -> list[RetrievedDoc]:
    return DEMO_KB.search(query, top_k=3)


# --------------------------------------------------------------------------- #
# ЭТАП 3 — ACT: генерация ответа (роутинг по сложности)
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM_PROMPT = """\
Ты — агент поддержки. Отвечай ТОЛЬКО на основании предоставленного контекста
из базы знаний. Если контекста недостаточно — явно скажи об этом, не
придумывай факты. В конце ответа перечисли id использованных документов
в формате: [источники: doc_id1, doc_id2].
"""


def generate_answer(model: str, query: str, context_docs: list[RetrievedDoc]) -> str:
    context_block = "\n\n".join(
        f"[{d.document.doc_id}] {d.document.title}\n{d.document.text}"
        for d in context_docs
    )
    user_prompt = f"Контекст:\n{context_block}\n\nВопрос пользователя:\n{query}"
    return _chat_text(model, ANSWER_SYSTEM_PROMPT, user_prompt)


# --------------------------------------------------------------------------- #
# ЭТАП 4 — OBSERVE: grounding-валидация
# --------------------------------------------------------------------------- #

VALIDATE_SYSTEM_PROMPT = """\
Ты — валидатор ответов поддержки. Тебе дан контекст и сгенерированный ответ.
Твоя задача: проверить, что КАЖДОЕ фактическое утверждение в ответе подтверждается контекстом.

Сначала проведи тщательный сравнительный анализ фактов из контекста и ответа.
Затем верни ТОЛЬКО JSON:
{ "grounded": true | false, "reason": "краткое объяснение" }
"""


def validate_grounding(model: str, answer: str, context_docs: list[RetrievedDoc]) -> dict[str, Any]:
    context_block = "\n\n".join(d.document.text for d in context_docs)
    user_prompt = f"Контекст:\n{context_block}\n\nОтвет для проверки:\n{answer}"
    return _chat_json(model, VALIDATE_SYSTEM_PROMPT, user_prompt)


# --------------------------------------------------------------------------- #
# Оркестрация цикла Reason -> Act -> Observe
# --------------------------------------------------------------------------- #

def run_agent(query: str) -> AgentResult:
    cycle_start = time.monotonic()
    seen_queries: set[str] = set()
    used_advanced_model = False

    # --- REASON (guardrail) ---
    classification = classify(query)
    if classification.get("risk") == "high" or classification.get("_parse_error"):
        return AgentResult(
            outcome=Outcome.ESCALATED,
            message="Передаю ваш запрос оператору поддержки.",
            escalation_reason="high_risk_or_classification_failed",
        )

    for iteration in range(1, MAX_ITERATIONS + 1):
        # --- Защита от Dead Loop: таймаут на весь цикл ---
        if time.monotonic() - cycle_start > TOTAL_CYCLE_TIMEOUT_SECONDS:
            return AgentResult(
                outcome=Outcome.ESCALATED,
                message="Не удалось обработать запрос вовремя, передаю оператору.",
                iterations_used=iteration,
                escalation_reason="cycle_timeout",
            )

        # --- Защита от Dead Loop: повтор идентичного запроса ---
        normalized = query.strip().lower()
        if normalized in seen_queries:
            return AgentResult(
                outcome=Outcome.ESCALATED,
                message="Не удалось найти ответ, передаю оператору.",
                iterations_used=iteration,
                escalation_reason="duplicate_query_detected",
            )
        seen_queries.add(normalized)

        # --- ACT: retrieval ---
        docs = retrieve_context(query)

        # --- OBSERVE: релевантность найденного ---
        if not docs or max(d.score for d in docs) < 0.15:
            return AgentResult(
                outcome=Outcome.ESCALATED,
                message="В базе знаний не нашлось релевантной информации, передаю оператору.",
                iterations_used=iteration,
                escalation_reason="low_retrieval_score",
            )

        # --- Защита от Dead Loop: лимит токенов диалога ---
        if _approx_tokens(query, *(d.document.text for d in docs)) > MAX_DIALOG_TOKENS_APPROX:
            return AgentResult(
                outcome=Outcome.ESCALATED,
                message="Запрос слишком объёмный для автоматической обработки, передаю оператору.",
                iterations_used=iteration,
                escalation_reason="token_budget_exceeded",
            )

        # --- REASON: выбор модели по сложности ---
        best_score = max(d.score for d in docs)
        is_simple = classification.get("complexity") == "simple" and best_score > 0.5
        
        # Выбираем модель для генерации
        model_for_answer = CHEAP_MODEL if (is_simple and not used_advanced_model) else ADVANCED_MODEL
        
        # Выбираем модель для валидации (используем ADVANCED_MODEL для сложных запросов
        # или если мы уже перешли на неё из-за провала валидации дешевой модели)
        model_for_validation = ADVANCED_MODEL if (classification.get("complexity") == "complex" or used_advanced_model) else CHEAP_MODEL

        # --- ACT: генерация ответа ---
        answer = generate_answer(model_for_answer, query, docs)

        # --- OBSERVE: grounding-валидация ---
        validation = validate_grounding(model_for_validation, answer, docs)

        if validation.get("grounded") is True:
            return AgentResult(
                outcome=Outcome.ANSWERED,
                message=answer,
                sources=[d.document.doc_id for d in docs],
                iterations_used=iteration,
            )

        # Если валидация провалена дешевой моделью, помечаем, что на следующей итерации нужно использовать ADVANCED_MODEL
        if not validation.get("grounded") and model_for_validation == CHEAP_MODEL:
            used_advanced_model = True
            
        continue

    return AgentResult(
        outcome=Outcome.ESCALATED,
        message="Не удалось сформировать проверяемый ответ, передаю оператору.",
        iterations_used=MAX_ITERATIONS,
        escalation_reason="grounding_validation_failed_max_iterations",
    )


# --------------------------------------------------------------------------- #
# Демо-запуск
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Агент поддержки готов ответить на ваши запросы. Введите запрос, 'exit' — выход.\n")
    while True:
        try:
            query = input("Запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"exit", "quit"}:
            break

        result = run_agent(query)
        print(f"\n[{result.outcome}] (итераций: {result.iterations_used})")
        print(result.message)
        if result.sources:
            print(f"Источники: {', '.join(result.sources)}")
        if result.escalation_reason:
            print(f"Причина эскалации: {result.escalation_reason}")
        print()


if __name__ == "__main__":
    main()

# QuasiHedge Bot — README

Интерактивный medium-quant Telegram-ассистент для оценки и мониторинга активов (акции + крипто). Этот репозиторий содержит полнофункциональный backend: сбор данных (Telegram/RSS/APIs), NLP-пайплайн, event-study, scoring, Monte-Carlo, риск-менеджмент и хранилище для исследований.

---

## Коротко — что умеет бот

* Сбор и дедупликация новостей из Telegram каналов и RSS
* Нормализация и NER → маппинг событий к тикерам
* NLP: классификация типа события, sentiment, relevance, embedding
* Event-study: поиск аналогов, подсчёт excess return и статистики (p-value)
* Многослойный скоринг (фундамент, рынок, события) → интерпретируемый score 0–10
* Monte-Carlo (GBM, bootstrap, regime switching) для оценки сценариев
* Position sizing (Kelly-подобный, волатильностный коррект) и VaR/CVaR
* Audit trail: все входные новости, версии моделей и отчёты сохраняются в БД
* Поддержка SQLite для локальной разработки и PostgreSQL для продакшена
* Готов к расширению: vector search (Faiss / PGVector), интеграция новых источников

---

## Быстрый старт

1. Скопируй репозиторий, создай виртуальное окружение и активируй его.
2. Создай файл `.env` в корне (у нас уже есть шаблон).
3. Заполни ключи: Telegram токен, `ALPHAVANTAGE_API_KEY` и т.д.
4. Инициализируй БД и заполни демо-данные (опционально):

```bash
python main.py init-db
python main.py seed-demo
python main.py check-config
python main.py run
```

---

## Важные файлы и структура (ключевые)

* `main.py` — точка входа (run | init-db | seed-demo | check-config)
* `app/config.py` — конфиг, парсер `.env` и валидация параметров
* `app/core.py` — оркестратор, обработка команд и flow (ingest → nlp → events → scoring → mc → risk)
* `app/db.py` — production-ready storage layer (SQLAlchemy 2.0)
* `app/services.py` — API-клиенты и ingestion (RSS / Telegram / Yahoo / AlphaVantage / CoinGecko)
* `app/nlp_events.py` — NLP pipeline и event matching
* `app/modeling.py` — scoring, ML hooks, explainability (SHAP)
* `app/mc_risk.py` — Monte-Carlo + risk sizing
* `data/` — БД, кеши, экспортируемые артефакты
* `models/` — обученные модели и векторизаторы
* `logs/` — лог-файлы

---

## Конфигурация / окружение

В `.env` указываются все ключи и параметры (DB URL, пороги, лимиты, Enable-флаги).
Ключевые параметры, которые используются модулем `db.py` и всем приложением:

* `DATABASE_URL` — SQLAlchemy URL (например `sqlite:///./data/store.sqlite` или `postgresql+psycopg2://user:pass@host/db`)
* `BASE_BENCHMARK` — тикер бенчмарка для excess returns (по умолчанию `SPY`)
* `ANALOG_LIMIT` — сколько кандидатов учитывать при поиске аналогов
* `EVENT_WINDOWS` — список горизонтов (например `30,90,180`)

> Советы: для разработки используйте SQLite; для деплоя — PostgreSQL (тот же код без правок).

---

## Зависимости (основные библиотеки)

В проекте используются современные инструменты:

* Python — runtime.
* Visual Studio Code — рекомендованный IDE.
* SQLAlchemy — ORM и engine.
* pandas и numpy — табличные операции и численная математика.
* scikit-learn и lightgbm — ML-модели.
* transformers / sentence-transformers — эмбеддинги и NLP.
* SciPy — статистика (p-value), опционально.
* yfinance, Alpha Vantage, CoinGecko — источники данных.
* Telegram + Telethon (или Pyrogram) — ingestion и команды.
* SQLite (dev) и PostgreSQL (prod).
* Для векторного поиска можно интегрировать Faiss или PGVector.

> Все зависимости перечислены в `requirements.txt`. Указывай версии в соответствии с твоим окружением.

---

## Архитектура: как данные проходят через систему

1. **Ingest** — `app/services.py` собирает новости/сообщения, выгружает цены и fundamentals.
2. **Preprocessing** — очистка, дедупликация, NER, маппинг тикеров.
3. **NLP** — `app/nlp_events.py` делает классификацию типа события, sentiment, relevance, строит embedding.
4. **Persist** — событие сохраняется в `events` в БД (`app/db.py`).
5. **Analogs / Event Study** — `app/db.find_analogs` ищет похожие события и `app/db.calculate_event_impact` рассчитывает excess returns и p-value.
6. **Scoring** — `app/modeling.py` объединяет фундамент, рынок и event-вклад в интерпретируемый score.
7. **Monte-Carlo + Risk** — `app/mc_risk.py` строит сценарии, выдаёт VaR/CVaR, вероятности и рекомендует размер позиции.
8. **Output** — `app/core.py` формирует Telegram-ответ: score, probabilities, confidence, топ-факторы, recommended position и ссылки на исходные события.

---

## Команды Telegram / CLI

Через Telegram бот (или CLI) доступны команды:

* `/start_analysis TICKER` — базовый сбор и скоринг
* `/add_info TICKER` — добавить список новостей/планов (batch)
* `/update_analysis TICKER` — пересчитать скоринг с новыми событиями
* `/simulate TICKER HORIZON` — Monte-Carlo на N месяцев
* `/compare_history TICKER` — найти исторические аналоги и статистику
* `/analyze_batch T1,T2,...` — пакетный анализ
* `/auto_watch TICKER on/off` — включить автослежение
* CLI: `python main.py init-db`, `python main.py seed-demo`, `python main.py run`, `python main.py check-config`

(Точные имена команд определены в `app/core.py` / `app/commands.py`.)

---

## Важные концепты и настройки качества

* **Anti-hallucination**: все causal claims подкрепляются статистикой (N, mean, p-value). Для confident claim требуется минимум `MIN_ANALOGS_REQUIRED` аналогов (настраивается).
* **Regime detection**: скорринг адаптируется под режим (bull / bear / crisis).
* **Explainability**: для каждого анализа возвращается top-5 факторов (SHAP-like) и уровень confidence.
* **Audit**: каждая сессия анализа сохраняется в `analyses` с ссылками на входные события и версией модели.
* **Safety**: рекомендации — вероятностные, с указанием риска и размера позиции; бот не обещает гарантий.

---

## Расширяемость и продакшен-готовность

* Переключение SQLite ↔ PostgreSQL происходит через `DATABASE_URL` в `.env` (без правок кода).
* Bulk upsert для цен реализован с использованием `ON CONFLICT` для PostgreSQL и оптимизированных путей для SQLite.
* Хранилище готово к интеграции с векторным поиском (Faiss / PGVector) для быстрого поиска аналогичных эмбеддингов.
* Модульность позволяет заменить источники данных, модели и политики риск-менеджмента без изменения интерфейса.

---

## Отладка и мониторинг

* Логи пишутся в `logs/bot.log` (ротация настроена в конфиге).
* `main.py check-config` выводит текущую конфигурацию (секреты маскируются).
* При критических изменениях скоринга система помечает запись как `REVIEW_REQUIRED` (human-in-loop).

---

## Частые вопросы

**Q: Как быстро получить рабочую версию?**
A: Используй SQLite и запусти `python main.py seed-demo`, затем `python main.py run`. Это развернёт базовую функциональность локально.

**Q: Нужны ли платные API?**
A: Нет — MVP работает на бесплатных потоках (`yfinance`, `CoinGecko`, RSS). Платные данные (options flow, Level2) улучшат качество, но не обязательны.

**Q: Как обучить ML-модели?**
A: Храни данные в `data/` и модели в `models/`. Скрипты обучения можно добавить в `scripts/train_*.py`, но начальная интеграция (LightGBM / XGBoost) уже учтена в `app/modeling.py`.

---

## Следующие шаги (рекомендации по развитию)

1. Подключить 10–15 качественных Telegram каналов (ручной отбор).
2. Заполнить историческую базу цен для интересующих тикеров (backfill).
3. Накопить события и начать walk-forward обучение event→reaction модели.
4. Добавить мониторинг и алерты на основе confidence/VaR.
5. При росте потребностей — вынести DB на managed PostgreSQL и добавить vector search (PGVector).

---

## Контакты / вклад

Если хочешь, я могу:

* сгенерировать `requirements.txt` c конкретными версиями;
* подготовить пример `app/core.py` с конкретной интеграцией команд;
* помочь выбрать 10–15 качественных Telegram каналов и RSS-источников для крипто и финанс.

---

Спасибо — теперь у тебя есть полноценный проект-каркас для **QuasiHedge Bot**: production-ready storage layer, мощная event-engine и probabilistic modeling, а также готовая дорожная карта для развития.

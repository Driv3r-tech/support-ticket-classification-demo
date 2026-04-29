# Support Ticket Classification Demo

Проект по обработке заявок поддержки и построению простого ML-классификатора.

## Цель проекта

Показать полный путь от чтения CSV/JSON-файлов до построения и оценки ML-модели для классификации заявок по категориям:
- bug
- feature
- support

## Что реализовано

- чтение данных из JSON/CSV;
- базовая валидация записей;
- анализ данных через pandas;
- baseline через DummyClassifier;
- preprocessing pipeline через TfidfVectorizer и OneHotEncoder;
- сравнение моделей:
  - DummyClassifier;
  - LogisticRegression;
  - LinearSVC;
  - MultinomialNB;
  - RandomForestClassifier;
- cross-validation через StratifiedKFold;
- проверка возможного data leakage;
- сравнение разных наборов признаков.

## Результаты

Baseline DummyClassifier:

| model | accuracy | macro_f1 |
|---|---:|---:|
| DummyClassifier | 0.40 | 0.19 |

Cross-validation:

| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std |
|---|---:|---:|---:|---:|
| RandomForestClassifier | 0.92 | 0.024 | 0.920 | 0.024 |
| LinearSVC | 0.91 | 0.037 | 0.905 | 0.040 |
| LogisticRegression | 0.80 | 0.045 | 0.790 | 0.049 |
| MultinomialNB | 0.70 | 0.032 | 0.665 | 0.056 |
| DummyClassifier | 0.40 | 0.000 | 0.190 | 0.000 |

## Leakage check

Были сравнены разные наборы признаков:

| feature_set | macro_f1_mean | macro_f1_std |
|---|---:|---:|
| message_only | 0.940 | 0.038 |
| message_user_priority | 0.930 | 0.025 |
| all_features | 0.920 | 0.024 |
| message_priority | 0.889 | 0.088 |

Вывод: лучший результат показал набор `message_only`. Это означает, что основной полезный сигнал содержится в тексте заявки, а дополнительные признаки могут добавлять шум.

## Главный вывод

Проект показывает не только обучение модели, но и базовую ML-логику:

```text
raw data → validation → DataFrame → baseline → pipeline → metrics → cross-validation → leakage check

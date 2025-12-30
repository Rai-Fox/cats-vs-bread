# Cats vs Bread

Фреймворк для обучения классификатора изображений «кот против хлеба».

Проект использует PyTorch Lightning, Hydra, DVC и MLflow для конфигурирования, управления данными и логирования экспериментов.

## Описание задачи

- **Идея**: отличать кота от хлеба на картинке. Потенциальный итоговый сервис — развлекательный TG-бот.
- **Формат данных**: вход — произвольное изображение, выход — вероятность каждого класса.
- **Метрики**: ROC-AUC, F1, accuracy. Метрики эксперимента логируются в MLflow.
- **Датасет**: Собран из:
  - https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
  - https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds
  - https://images.cv/download/bread/2456

## Структура проекта

- `cats_vs_bread/` — основной код проекта:
  - `models/` — модели и датамодули PyTorch Lightning.
    - `lightning_module.py` — lightning-модуль с моделью, шагами обучения и валидации.
    - `data_module.py` — lightning-датамодуль для загрузки и препроцессинга данных.
    - `model.py` — модель для классификации.
  - `utils/` — утилиты
    - `logging_utils.py` — функции для логирования выполнения кода.
  - `configs.py` — конфигурации Hydra.
  - `train.py` — скрипт для обучения модели.
- `cats_vs_bread.py` — точка входа для запуска команд через Hydra.
- `configs/` — конфигурации Hydra для экспериментов.
- `data/` — каталог для хранения данных с помощью DVC.
- `docker-compose.yml` — конфигурация Docker Compose для запуска MLflow-сервера.

## Быстрый старт (Setup)

1. Установите зависимости:

   Предлагаются версии с поддержкой CUDA12.8 и без. Выберите подходящую:
   - С поддержкой CUDA
     ```bash
     uv sync --extra cu128
     ```
   - Без поддержки CUDA
     ```bash
     uv sync --extra cpu
     ```

2. Запустите MLflow-сервер (по умолчанию на `http://127.0.0.1:5000`):
   ```bash
   docker-compose up --build
   ```

## Обучение (Train)

Для обучения модели выполните команду:

```bash
uv run cats_vs_bread.py train.max_epochs=3
```

Можно переопределить гиперпараметры через CLI. Например:

- Эпохи: `train.max_epochs=5`
- Размер батча: `data.batch_size=16`
- MLflow: `logger.tracking_url=http://127.0.0.1:8080`

## Качество кода

- Для проверки качество кода используйте:

```bash
uv run --extra dev pre-commit run -a
```

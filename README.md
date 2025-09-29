# Подготовка виртуальной машины

## Склонируйте репозиторий

Склонируйте репозиторий проекта:

```
git clone https://github.com/arigatory/mle-project-sprint-4-v001.git
```

## Активируйте виртуальное окружение

Используйте то же самое виртуальное окружение, что и созданное для работы с уроками. Если его не существует, то его следует создать.

Создать новое виртуальное окружение можно командой:

```
python3 -m venv env_recsys_start
```

После его инициализации следующей командой

```
. env_recsys_start/bin/activate
```

установите в него необходимые Python-пакеты следующей командой

```
pip install -r requirements.txt
```

### Скачайте файлы с данными

Для начала работы понадобится три файла с данными:
- [tracks.parquet](https://storage.yandexcloud.net/mle-data/ym/tracks.parquet)
- [catalog_names.parquet](https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet)
- [interactions.parquet](https://storage.yandexcloud.net/mle-data/ym/interactions.parquet)
 
Скачайте их в директорию локального репозитория. Для удобства вы можете воспользоваться командой wget:

```
wget https://storage.yandexcloud.net/mle-data/ym/tracks.parquet

wget https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet

wget https://storage.yandexcloud.net/mle-data/ym/interactions.parquet
```

## Запустите Jupyter Lab

Запустите Jupyter Lab в командной строке

```
jupyter lab --ip=0.0.0.0 --no-browser
```

# Расчёт рекомендаций

Код для выполнения первой части проекта находится в файле `recommendations.ipynb`. Изначально, это шаблон. Используйте его для выполнения первой части проекта.

# Сервис рекомендаций

Код сервиса рекомендаций находится в файле `recommendations_service.py`.

## Запуск сервиса рекомендаций

1. Убедитесь, что виртуальное окружение активировано:
```bash
source env_recsys_start/bin/activate
```

2. Запустите сервис:
```bash
python recommendations_service.py
```

Сервис будет доступен по адресу: http://localhost:8000

## API Endpoints

- `GET /` - проверка работоспособности сервиса
- `POST /recommend` - получение рекомендаций для пользователя
- `POST /track_interaction` - отслеживание взаимодействия пользователя с треком
- `GET /user_history/{user_id}` - получение истории пользователя
- `GET /stats` - статистика сервиса

# Пошаговая проверка работоспособности проекта

## Шаг 1: Проверить структуру проекта
```bash
ls -la
```
Должны быть видны файлы: `recommendations_service.py`, `test_service.py`, `requirements.txt`, и parquet файлы.

## Шаг 2: Активировать виртуальное окружение
```bash
source env_recsys_start/bin/activate
```

## Шаг 3: Проверить зависимости
```bash
pip list | grep -E "(fastapi|pandas|uvicorn)"
```
Должны быть установлены основные пакеты.

## Шаг 4: Проверить наличие файлов данных
```bash
ls -la *.parquet
```
Должны быть файлы: `tracks.parquet`, `interactions.parquet`, `catalog_names.parquet`.

## Шаг 5: Запустить сервис рекомендаций
```bash
python recommendations_service.py
```
Должно появиться сообщение о том, что сервис запущен на порту 8000.

## Шаг 6: Проверить работу сервиса (в новом терминале)
```bash
curl http://localhost:8000/
```
Должен вернуться JSON с сообщением о работающем сервисе.

## Шаг 7: Запустить автотесты
```bash
python test_service.py
```
Все 3 теста должны пройти успешно.

# Инструкции для тестирования сервиса

Код для тестирования сервиса находится в файле `test_service.py`.

## Автоматическое тестирование

Запустите автотесты командой:
```bash
python test_service.py
```

Тесты проверяют три сценария:
1. **Новый пользователь** (без истории) - должен получать только популярные рекомендации
2. **Пользователь с офлайн историей** - должен получать микс коллаборативных и контентных рекомендаций
3. **Пользователь с онлайн историей** - должен получать микс всех типов рекомендаций

## Ручное тестирование API

### Получение рекомендаций
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 123, "num_recommendations": 5}'
```

### Отслеживание взаимодействия
```bash
curl -X POST "http://localhost:8000/track_interaction?user_id=123&track_id=456"
```

### Получение истории пользователя
```bash
curl "http://localhost:8000/user_history/123"
```

### Получение статистики
```bash
curl "http://localhost:8000/stats"
```

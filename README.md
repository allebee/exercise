# Exercise Pose Detection

Этот проект использует FastAPI и MediaPipe для обнаружения и анализа различных упражнений на основе видео потока.


## Установка

1. Клонируйте репозиторий:
    ```sh
    git clone <URL вашего репозитория>
    cd <название репозитория>
    ```

2. Создайте и активируйте виртуальное окружение:
    ```sh
    python -m venv venv
    source venv/bin/activate  # Для Windows используйте `venv\Scripts\activate`
    ```

3. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

## Запуск проекта

### Локально

1. Запустите сервер:
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 8011
    ```

2. Откройте браузер и перейдите по адресу `http://localhost:8011`.

### В Docker

1. Постройте Docker образ:
    ```sh
    docker build -t exercise-pose-detection .
    ```

2. Запустите контейнер:
    ```sh
    docker run -p 8011:8011 exercise-pose-detection
    ```

## Использование

На главной странице вы можете выбрать одно из упражнений для анализа:
- Plank Detection
- Lunge Detection
- Bicep Curl Detection
- Squat Detection

После выбора упражнения, веб-камера начнет захват видео, и результаты анализа будут отображаться в реальном времени.

## Структура файлов

- [main.py](http://_vscodecontentref_/14): Основной файл приложения FastAPI.
- [bicep_curl.py](http://_vscodecontentref_/15), [lunge.py](http://_vscodecontentref_/16), [plank.py](http://_vscodecontentref_/17), [squat.py](http://_vscodecontentref_/18): Скрипты для анализа различных упражнений.
- [bicep_inference.py](http://_vscodecontentref_/19), [lunge_inference.py](http://_vscodecontentref_/20), [plank_inference.py](http://_vscodecontentref_/21), [squat_inference.py](http://_vscodecontentref_/22): Веб-сокеты для обработки видео потока.
- [utils.py](http://_vscodecontentref_/23): Вспомогательные функции для анализа поз.
- [static](http://_vscodecontentref_/24): Статические файлы (HTML, CSS, JS).
- [templates](http://_vscodecontentref_/25): Шаблоны HTML.
- [model](http://_vscodecontentref_/26): Модели машинного обучения для анализа упражнений.
- [requirements.txt](http://_vscodecontentref_/27): Зависимости проекта.
- [dockerfile](http://_vscodecontentref_/28): Docker файл для создания контейнера.


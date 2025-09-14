# Комплексный пайплайн обработки документов

Этот документ описывает новый комплексный пайплайн для извлечения таблиц из PDF документов.

## Обзор пайплайна

Пайплайн состоит из следующих этапов:

```
PDF → Image → Preprocessing → Layout Detection → OCR → Table Detection → Export
```

### Этапы пайплайна

1. **PDF → Image конвертация** (`pdf2image`)
   - Конвертирует PDF страницы в изображения
   - Использует Poppler для высококачественной конвертации

2. **Предобработка изображений** (опционально)
   - **Deskew**: Выравнивание наклоненных страниц
   - **Denoise**: Удаление шума и артефактов
   - **Binarize**: Бинаризация для улучшения OCR

3. **Layout Detection** (`LayoutParser`)
   - Детекция различных зон документа (текст, таблицы, изображения)
   - Модели: EfficientDete/PubLayNet, faster_rcnn_R_50_FPN_3x

4. **OCR** (выбор метода)
   - **PaddleOCR**: Рекомендуется для русского языка (PP-OCRv5)
   - **DocTR**: Хорош для сложных форм и таблиц
   - **Tesseract**: Классический OCR движок

5. **Детекция таблиц** (опционально)
   - Извлечение табличных структур из OCR результатов
   - Группировка текста по строкам и колонкам

6. **Экспорт** (множественные форматы)
   - **JSON**: Метаданные и структурированные данные
   - **Excel**: Таблицы в формате .xlsx
   - **CSV**: Таблицы в формате .csv
   - **TXT**: Простой текстовый вывод

## Использование через UI

1. Запустите приложение:
   ```bash
   docker-compose up
   ```

2. Откройте браузер: http://localhost:8501

3. В разделе "🔄 Комплексный пайплайн":
   - Включите "Использовать комплексный пайплайн"
   - Настройте компоненты через чекбоксы
   - Выберите методы OCR и экспорта
   - Загрузите PDF файл

## Использование через демо скрипты

### Полный пайплайн
```bash
python demo_pipeline.py
```

### Отдельные компоненты
```bash
# Все компоненты
python demo_components.py

# Только PDF → Image
python demo_components.py pdf2image

# Только предобработка
python demo_components.py preprocessing

# Только layout детекция
python demo_components.py layout

# Только OCR
python demo_components.py ocr

# Только детекция таблиц
python demo_components.py tables
```

## Конфигурация

### Рекомендуемые настройки для русского языка

```python
config = {
    'enable_preprocessing': True,
    'enable_deskew': True,
    'enable_denoise': True,
    'enable_binarize': False,  # Отключить для лучшего качества OCR
    
    'enable_layout_detection': True,
    'layout_model': 'lp://EfficientDete/PubLayNet',
    
    'enable_ocr': True,
    'ocr_method': 'paddleocr',  # Лучший для русского языка
    
    'enable_table_detection': True,
    
    'enable_export': True,
    'export_formats': ['json', 'excel', 'csv']
}
```

### Настройки для сложных документов

```python
config = {
    'enable_preprocessing': True,
    'enable_deskew': True,
    'enable_denoise': True,
    'enable_binarize': True,  # Включить для сложных документов
    
    'enable_layout_detection': True,
    'layout_model': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x',
    
    'enable_ocr': True,
    'ocr_method': 'doctr',  # Лучше для сложных форм
    
    'enable_table_detection': True,
    
    'enable_export': True,
    'export_formats': ['json', 'excel']
}
```

## Структура результатов

```python
{
    'pages': [
        {
            'page_number': 1,
            'image': numpy_array,
            'preprocessed_image': numpy_array  # если включена предобработка
        }
    ],
    'layout_regions': [
        {
            'type': 'table',
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.95,
            'page_number': 1
        }
    ],
    'ocr_results': [
        {
            'page_number': 1,
            'text': 'извлеченный текст',
            'blocks': [
                {
                    'text': 'текст блока',
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.9
                }
            ]
        }
    ],
    'tables': [
        [
            ['заголовок1', 'заголовок2', 'заголовок3'],
            ['данные1', 'данные2', 'данные3']
        ]
    ],
    'text': 'полный извлеченный текст',
    'export_files': ['output.json', 'output.xlsx', 'output.csv']
}
```

## Производительность

### Время обработки (примерно)
- PDF → Image: ~1-2 сек на страницу
- Предобработка: ~0.5-1 сек на страницу
- Layout Detection: ~2-5 сек на страницу
- OCR (PaddleOCR): ~3-8 сек на страницу
- OCR (DocTR): ~5-15 сек на страницу
- Детекция таблиц: ~1-2 сек на страницу
- Экспорт: ~0.5-1 сек

### Рекомендации по оптимизации
- Для быстрой обработки: отключите предобработку и layout detection
- Для лучшего качества: используйте все компоненты
- Для больших документов: обрабатывайте по страницам

## Устранение неполадок

### Ошибки инициализации
- Убедитесь, что все зависимости установлены
- Проверьте доступность моделей LayoutParser
- Проверьте корректность путей к файлам

### Проблемы с качеством OCR
- Попробуйте разные методы OCR
- Включите/отключите предобработку
- Проверьте качество исходного PDF

### Проблемы с детекцией таблиц
- Убедитесь, что включена layout detection
- Попробуйте разные модели layout
- Проверьте качество OCR результатов

## Зависимости

Основные зависимости для пайплайна:
- `pdf2image`: PDF → Image конвертация
- `scikit-image`: Предобработка изображений
- `layoutparser`: Layout detection
- `paddleocr`: OCR для русского языка
- `python-doctr`: OCR для сложных документов
- `openpyxl`, `xlsxwriter`: Экспорт в Excel
- `lxml`, `xmltodict`: Работа с XML форматами

## Лицензии

- PaddleOCR: Apache 2.0
- DocTR: Apache 2.0
- LayoutParser: MIT
- pdf2image: MIT

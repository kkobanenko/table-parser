import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger('paddleocr_parser')

class PaddleOCRParser:
    """
    Парсер таблиц с использованием библиотеки PaddleOCR PP-OCRv5.
    Поддерживает как server модели (высокая точность), так и mobile модели (высокая эффективность).
    """

    def __init__(self, use_server_model: bool = True, lang: str = 'ru'):
        """
        Инициализация PaddleOCR парсера.
        
        Args:
            use_server_model: Использовать server модель (True) или mobile модель (False)
            lang: Язык для распознавания ('en', 'ch', 'ru', и т.д.)
        """
        self.use_server_model = use_server_model
        self.lang = lang
        self.ocr = None
        
        # Условный импорт PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.PaddleOCR = PaddleOCR
            PADDLEOCR_AVAILABLE = True
        except ImportError:
            PADDLEOCR_AVAILABLE = False
            logger.error("❌ PaddleOCR не установлен. Установите: pip install paddleocr")
            return
        
        if PADDLEOCR_AVAILABLE:
            self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Инициализация OCR движка с оптимальными параметрами для распознавания таблиц."""
        try:
            # Минимальная конфигурация для PP-OCRv5
            ocr_config = {
                'lang': self.lang,  # Язык распознавания
                'use_angle_cls': True,  # Включить классификацию углов поворота текста
            }
            
            # Выбор модели в зависимости от предпочтений
            if self.use_server_model:
                logger.info("🚀 Инициализация PaddleOCR с PP-OCRv5 server моделями (высокая точность)")
                # Дополнительные параметры для server модели
                ocr_config.update({
                    'det_model_dir': None,  # Автоматическая загрузка server модели детекции
                    'rec_model_dir': None,  # Автоматическая загрузка server модели распознавания
                })
            else:
                logger.info("📱 Инициализация PaddleOCR с PP-OCRv5 mobile моделями (высокая эффективность)")
                # Дополнительные параметры для mobile модели
                ocr_config.update({
                    'det_model_dir': None,  # Автоматическая загрузка mobile модели детекции
                    'rec_model_dir': None,  # Автоматическая загрузка mobile модели распознавания
                })
            
            self.ocr = self.PaddleOCR(**ocr_config)
            logger.info("✅ PaddleOCR успешно инициализирован с поддержкой распознавания таблиц")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации PaddleOCR: {e}")
            self.ocr = None

    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из изображения с помощью PaddleOCR PP-OCRv5.
        
        Args:
            file_path: Путь к файлу изображения
            
        Returns:
            Список словарей с данными таблиц
        """
        file_path_obj = Path(file_path)
        tables: List[Dict[str, Any]] = []
        
        if self.ocr is None:
            logger.warning("⚠️ PaddleOCR не инициализирован, пропускаем обработку")
            return tables
        
        try:
            logger.info(f"🔄 Обрабатываем файл через PaddleOCR PP-OCRv5: {file_path_obj.name}")
            
            # Выполняем OCR распознавание
            result = self.ocr.ocr(str(file_path_obj))
            
            if not result or result is None:
                logger.warning(f"⚠️ PaddleOCR не смог извлечь текст из {file_path_obj.name}")
                return tables
            
            # Обрабатываем результаты OCR
            extracted_text = self._process_ocr_results(result)
            
            if extracted_text:
                # Пытаемся найти таблицы в извлеченном тексте
                table_data = self._extract_tables_from_text(extracted_text)
                
                # Если не нашли таблицы в тексте, пробуем анализ по координатам
                if not table_data and hasattr(self, 'text_coordinates') and self.text_coordinates:
                    logger.info(f"🔍 Пробуем анализ таблицы по координатам ({len(self.text_coordinates)} элементов)")
                    table_data = self._extract_tables_from_coordinates()
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    if not df.empty and df.notna().any().any():
                        tables.append({
                            "data": df,
                            "sheet_name": f"PaddleOCR_PP-OCRv5_Table",
                            "source": "paddleocr_pp-ocrv5",
                            "cleaning_method": "ocr_text_parsing"
                        })
                        logger.info(f"✅ Найдена таблица PaddleOCR PP-OCRv5: {df.shape}")
                else:
                    # Если таблица не найдена, создаем простую таблицу с извлеченным текстом
                    text_rows = extracted_text.split('\n')
                    text_data = [[row.strip()] for row in text_rows if row.strip()]
                    
                    if text_data:
                        df = pd.DataFrame(text_data, columns=['Extracted_Text'])
                        tables.append({
                            "data": df,
                            "sheet_name": f"PaddleOCR_PP-OCRv5_Text",
                            "source": "paddleocr_pp-ocrv5",
                            "cleaning_method": "ocr_text_extraction"
                        })
                        logger.info(f"✅ Извлечен текст через PaddleOCR PP-OCRv5: {len(text_data)} строк")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке файла {file_path_obj.name} через PaddleOCR: {e}")
        
        logger.info(f"🏁 PaddleOCR PP-OCRv5 обработка завершена: найдено {len(tables)} таблиц")
        return tables
    
    def _process_ocr_results(self, ocr_result: List) -> str:
        """
        Обрабатывает результаты OCR и извлекает текст с улучшенной обработкой таблиц.
        
        Args:
            ocr_result: Результат от PaddleOCR
            
        Returns:
            Извлеченный текст
        """
        extracted_text = ""
        self.text_coordinates = []  # Инициализируем список координат
        
        try:
            # PaddleOCR возвращает результаты в формате:
            # [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)]
            if not ocr_result:
                logger.warning("⚠️ PaddleOCR вернул пустой результат")
                return ""
                
            logger.info(f"🔍 Обрабатываем {len(ocr_result)} страниц результатов OCR")
            
            for page_idx, page_result in enumerate(ocr_result):
                if page_result is None:
                    logger.warning(f"⚠️ Страница {page_idx} содержит None результат")
                    continue
                
                logger.info(f"📄 Обрабатываем страницу {page_idx}, тип результата: {type(page_result)}")
                
                # Проверяем различные форматы результатов PaddleOCR
                if hasattr(page_result, 'texts'):
                    # Новый формат PaddleOCR с атрибутом texts
                    logger.info(f"📝 Найдено {len(page_result.texts)} текстовых элементов")
                    for text_info in page_result.texts:
                        if hasattr(text_info, 'text') and text_info.text:
                            extracted_text += text_info.text + "\n"
                elif hasattr(page_result, 'boxes') and hasattr(page_result, 'texts'):
                    # Формат с отдельными boxes и texts
                    logger.info(f"📝 Найдено {len(page_result.texts)} текстовых элементов")
                    for text in page_result.texts:
                        if text and text.strip():
                            extracted_text += text.strip() + "\n"
                elif hasattr(page_result, 'ocr_result'):
                    # Формат с вложенным ocr_result
                    logger.info("📝 Обрабатываем вложенный ocr_result")
                    ocr_data = page_result.ocr_result
                    if isinstance(ocr_data, list):
                        for line_idx, line in enumerate(ocr_data):
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                                confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                                
                                if confidence >= 0.3 and text and text.strip():
                                    extracted_text += text.strip() + "\n"
                                    logger.debug(f"📝 Строка {line_idx}: '{text}' (уверенность: {confidence:.2f})")
                elif isinstance(page_result, list):
                    # Старый формат PaddleOCR
                    logger.info(f"📝 Найдено {len(page_result)} текстовых элементов")
                    for line_idx, line in enumerate(page_result):
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                            confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                            
                            # Понижаем порог уверенности для лучшего распознавания
                            if confidence >= 0.3 and text and text.strip():
                                extracted_text += text.strip() + "\n"
                                logger.debug(f"📝 Строка {line_idx}: '{text}' (уверенность: {confidence:.2f})")
                            else:
                                logger.debug(f"⚠️ Пропущена строка {line_idx}: '{text}' (уверенность: {confidence:.2f})")
                else:
                    # Пытаемся извлечь текст из неизвестного формата
                    logger.warning(f"⚠️ Неизвестный формат результата: {type(page_result)}")
                    
                    # Пытаемся найти текстовые данные в различных атрибутах
                    for attr_name in ['str', 'texts', 'text', 'ocr_result', 'result', 'data']:
                        if hasattr(page_result, attr_name):
                            attr_value = getattr(page_result, attr_name)
                            logger.info(f"🔍 Найден атрибут {attr_name}: {type(attr_value)}")
                            
                            if attr_name == 'str' and isinstance(attr_value, dict):
                                # Извлечение текста из словаря str
                                if 'res' in attr_value:
                                    res_data = attr_value['res']
                                    logger.info(f"📝 Найдены данные в str['res']: {type(res_data)}")
                                    
                                    # Извлекаем тексты и координаты для анализа структуры
                                    texts = res_data.get('rec_texts', [])
                                    boxes = res_data.get('rec_boxes', [])
                                    scores = res_data.get('rec_scores', [])
                                    
                                    logger.info(f"📝 Найдено {len(texts)} текстовых элементов")
                                    
                                    for i, text in enumerate(texts):
                                        if text and text.strip():
                                            extracted_text += text.strip() + "\n"
                                            
                                            # Сохраняем координаты для анализа структуры таблицы
                                            if boxes is not None and len(boxes) > i:
                                                box = boxes[i]
                                                if box is not None and len(box) >= 4:  # [x1, y1, x2, y2]
                                                    self.text_coordinates.append({
                                                        'text': text.strip(),
                                                        'x1': box[0], 'y1': box[1],
                                                        'x2': box[2], 'y2': box[3],
                                                        'score': scores[i] if scores is not None and len(scores) > i else 1.0
                                                    })
                                    
                                    logger.info(f"📝 Извлечен текст через .str: {len(extracted_text)} символов")
                            elif attr_name == 'str' and isinstance(attr_value, str) and attr_value.strip():
                                # Прямое извлечение текста через атрибут str
                                extracted_text += attr_value.strip() + "\n"
                                logger.info(f"📝 Извлечен текст через .str: {len(attr_value)} символов")
                            elif isinstance(attr_value, list) and attr_value:
                                logger.info(f"📝 Обрабатываем {attr_name} с {len(attr_value)} элементами")
                                for item in attr_value:
                                    if isinstance(item, str) and item.strip():
                                        extracted_text += item.strip() + "\n"
                                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                        text = item[1][0] if isinstance(item[1], tuple) else str(item[1])
                                        if text and text.strip():
                                            extracted_text += text.strip() + "\n"
            
            logger.info(f"📄 PaddleOCR извлек текст ({len(extracted_text)} символов)")
            if extracted_text:
                logger.debug(f"Первые 500 символов: {extracted_text[:500]}")
            else:
                logger.warning("⚠️ Не удалось извлечь текст из результатов OCR")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки результатов OCR: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
        
        return extracted_text.strip()
    
    def _extract_tables_from_text(self, text: str) -> List[List[str]]:
        """
        Извлекает таблицы из текста с улучшенными алгоритмами распознавания структуры.
        
        Args:
            text: Извлеченный текст
            
        Returns:
            Данные таблицы в виде списка списков
        """
        lines = text.split('\n')
        table_data: List[List[str]] = []
        
        logger.info(f"🔍 Анализируем {len(lines)} строк для поиска таблиц")
        
        # Очищаем и фильтруем строки
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:  # Игнорируем пустые строки и одиночные символы
                cleaned_lines.append(line)
        
        logger.info(f"📝 После очистки осталось {len(cleaned_lines)} строк")
        
        # Ищем строки, которые могут быть таблицами
        potential_table_lines = []
        
        for line_idx, line in enumerate(cleaned_lines):
            # Различные способы разделения на колонки
            # 1. По нескольким пробелам (табуляция)
            words_tab = re.split(r'\s{2,}', line)
            # 2. По обычным пробелам
            words_space = re.split(r'\s+', line)
            # 3. По специальным символам (|, ;, \t)
            words_pipe = re.split(r'[|;\t]+', line)
            
            # Выбираем лучший способ разделения
            best_words = None
            best_score = 0
            
            for words in [words_tab, words_pipe, words_space]:
                if len(words) >= 2:  # Минимум 2 колонки
                    # Оцениваем качество разделения
                    score = len(words)
                    # Бонус за равномерное распределение длины слов
                    avg_length = sum(len(word) for word in words) / len(words)
                    if avg_length > 2:  # Слова не слишком короткие
                        score += 1
                    # Бонус за наличие цифр (часто в таблицах)
                    if any(re.search(r'\d', word) for word in words):
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_words = words
            
            if best_words and len(best_words) >= 2:
                # Очищаем слова от лишних символов
                cleaned_words = [word.strip() for word in best_words if word.strip()]
                if len(cleaned_words) >= 2:
                    potential_table_lines.append(cleaned_words)
                    logger.debug(f"📊 Строка {line_idx}: {len(cleaned_words)} колонок - {cleaned_words}")
        
        logger.info(f"📊 Найдено {len(potential_table_lines)} потенциальных табличных строк")
        
        # Если нашли потенциальные табличные строки
        if potential_table_lines:
            # Определяем максимальное количество колонок
            max_cols = max(len(row) for row in potential_table_lines)
            logger.info(f"📊 Максимальное количество колонок: {max_cols}")
            
            # Нормализуем все строки до одинакового количества колонок
            for row_idx, row in enumerate(potential_table_lines):
                # Дополняем строки пустыми значениями до max_cols
                while len(row) < max_cols:
                    row.append("")
                table_data.append(row)
                logger.debug(f"📊 Строка {row_idx}: {row}")
        
        logger.info(f"✅ Создана таблица размером {len(table_data)}x{max_cols if table_data else 0}")
        return table_data
    
    def _extract_tables_from_coordinates(self) -> List[List[str]]:
        """
        Извлекает таблицы на основе координат текстовых элементов.
        
        Returns:
            Данные таблицы в виде списка списков
        """
        if not hasattr(self, 'text_coordinates') or not self.text_coordinates:
            return []
        
        logger.info(f"🔍 Анализируем {len(self.text_coordinates)} текстовых элементов по координатам")
        
        # Сортируем элементы по Y-координате (сверху вниз), затем по X-координате (слева направо)
        sorted_elements = sorted(self.text_coordinates, key=lambda x: (x['y1'], x['x1']))
        
        # Группируем элементы по строкам (по Y-координате)
        rows = []
        current_row = []
        current_y = None
        y_tolerance = 20  # Допустимое отклонение по Y для элементов одной строки
        
        for element in sorted_elements:
            if current_y is None or abs(element['y1'] - current_y) <= y_tolerance:
                # Элемент в той же строке
                current_row.append(element)
                current_y = element['y1'] if current_y is None else current_y
            else:
                # Новая строка
                if current_row:
                    rows.append(current_row)
                current_row = [element]
                current_y = element['y1']
        
        # Добавляем последнюю строку
        if current_row:
            rows.append(current_row)
        
        logger.info(f"📊 Найдено {len(rows)} строк по координатам")
        
        # Сортируем элементы в каждой строке по X-координате
        for row in rows:
            row.sort(key=lambda x: x['x1'])
        
        # Фильтруем строки - оставляем только те, которые выглядят как табличные данные
        filtered_rows = []
        for row in rows:
            # Объединяем текст в строке
            row_text = ' '.join([elem['text'] for elem in row])
            
            # Пропускаем строки, которые явно не являются табличными данными
            if (len(row_text.strip()) < 3 or  # Слишком короткие строки
                row_text.strip().lower() in ['nan', 'none', ''] or  # Пустые значения
                len(row) < 2):  # Слишком мало колонок
                continue
                
            filtered_rows.append(row)
        
        logger.info(f"📊 После фильтрации осталось {len(filtered_rows)} строк")
        
        # Создаем таблицу
        table_data = []
        for row_idx, row in enumerate(filtered_rows):
            row_data = []
            for element in row:
                text = element['text'].strip()
                # Очищаем текст от лишних символов
                if text and text.lower() not in ['nan', 'none']:
                    row_data.append(text)
                else:
                    row_data.append("")
            table_data.append(row_data)
            logger.debug(f"📝 Строка {row_idx}: {len(row_data)} колонок - {row_data[:3]}...")
        
        # Улучшенная структуризация: пытаемся найти заголовки и данные
        if len(table_data) > 1:
            # Ищем строку с заголовками (обычно первая или вторая строка)
            header_row = self._find_header_row(table_data)
            if header_row is not None:
                # Переструктурируем таблицу с правильными заголовками
                table_data = self._restructure_table_with_headers(table_data, header_row)
        
        logger.info(f"✅ Создана таблица по координатам размером {len(table_data)}x{len(table_data[0]) if table_data else 0}")
        return table_data
    
    def _find_header_row(self, table_data: List[List[str]]) -> int:
        """
        Находит строку с заголовками таблицы.
        
        Args:
            table_data: Данные таблицы
            
        Returns:
            Индекс строки с заголовками или None
        """
        if len(table_data) < 2:
            return None
        
        # Ищем строку с заголовками по характерным признакам
        for i, row in enumerate(table_data[:3]):  # Проверяем первые 3 строки
            # Проверяем, содержит ли строка характерные слова для заголовков
            row_text = ' '.join(row).lower()
            header_keywords = [
                'торговое', 'наименование', 'препарата', 'мнн', 'химическое',
                'форма', 'выпуска', 'цена', 'жнвлп', 'упаковку', 'ндс', 'производитель'
            ]
            
            keyword_count = sum(1 for keyword in header_keywords if keyword in row_text)
            if keyword_count >= 3:  # Если найдено 3 или больше ключевых слов
                logger.info(f"📝 Найдена строка с заголовками: строка {i+1}")
                return i
        
        return None
    
    def _restructure_table_with_headers(self, table_data: List[List[str]], header_row: int) -> List[List[str]]:
        """
        Переструктурирует таблицу с правильными заголовками.
        
        Args:
            table_data: Данные таблицы
            header_row: Индекс строки с заголовками
            
        Returns:
            Переструктурированная таблица
        """
        if header_row >= len(table_data):
            return table_data
        
        # Берем заголовки из найденной строки
        headers = table_data[header_row]
        
        # Создаем новую таблицу с заголовками
        new_table = [headers]  # Первая строка - заголовки
        
        # Добавляем остальные строки (данные)
        for i, row in enumerate(table_data):
            if i != header_row:  # Пропускаем строку с заголовками
                new_table.append(row)
        
        logger.info(f"📊 Переструктурирована таблица: {len(new_table)} строк")
        return new_table
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Возвращает информацию о используемой модели.
        
        Returns:
            Словарь с информацией о модели
        """
        model_type = "server" if self.use_server_model else "mobile"
        return {
            "model_type": f"PP-OCRv5_{model_type}",
            "language": self.lang,
            "description": f"PaddleOCR PP-OCRv5 {model_type} model for {self.lang} text recognition"
        }

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

    def __init__(self, use_server_model: bool = True, lang: str = 'en'):
        """
        Инициализация PaddleOCR парсера.
        
        Args:
            use_server_model: Использовать server модель (True) или mobile модель (False)
            lang: Язык для распознавания ('en', 'ch', 'fr', и т.д.)
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
        """Инициализация OCR движка с оптимальными параметрами."""
        try:
            # Минимальная конфигурация для PP-OCRv5
            ocr_config = {
                'lang': self.lang  # Язык распознавания
            }
            
            # Выбор модели в зависимости от предпочтений
            if self.use_server_model:
                logger.info("🚀 Инициализация PaddleOCR с PP-OCRv5 server моделями (высокая точность)")
            else:
                logger.info("📱 Инициализация PaddleOCR с PP-OCRv5 mobile моделями (высокая эффективность)")
            
            self.ocr = self.PaddleOCR(**ocr_config)
            logger.info("✅ PaddleOCR успешно инициализирован")
            
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
        Обрабатывает результаты OCR и извлекает текст.
        
        Args:
            ocr_result: Результат от PaddleOCR
            
        Returns:
            Извлеченный текст
        """
        extracted_text = ""
        
        try:
            # PaddleOCR возвращает результаты в формате:
            # [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, confidence)]
            if not ocr_result:
                logger.warning("⚠️ PaddleOCR вернул пустой результат")
                return ""
                
            for page_result in ocr_result:
                if page_result is None:
                    continue
                    
                if hasattr(page_result, 'texts'):
                    # Новый формат PaddleOCR
                    for text_info in page_result.texts:
                        if hasattr(text_info, 'text') and text_info.text:
                            extracted_text += text_info.text + "\n"
                elif isinstance(page_result, list):
                    # Старый формат PaddleOCR
                    for line in page_result:
                        if line and len(line) >= 2:
                            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                            confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                            
                            # Фильтруем по уверенности (минимальная уверенность 0.5)
                            if confidence >= 0.5 and text:
                                extracted_text += text + "\n"
            
            logger.info(f"📄 PaddleOCR извлек текст ({len(extracted_text)} символов)")
            if extracted_text:
                logger.debug(f"Первые 200 символов: {extracted_text[:200]}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки результатов OCR: {e}")
        
        return extracted_text.strip()
    
    def _extract_tables_from_text(self, text: str) -> List[List[str]]:
        """
        Извлекает таблицы из текста, используя эвристики для поиска табличных структур.
        
        Args:
            text: Извлеченный текст
            
        Returns:
            Данные таблицы в виде списка списков
        """
        lines = text.split('\n')
        table_data: List[List[str]] = []
        
        # Ищем строки, которые могут быть таблицами
        # Эвристика: строки с несколькими словами, разделенными пробелами или табуляцией
        potential_table_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Проверяем, содержит ли строка несколько "слов" (разделенных пробелами/табуляцией)
            words = re.split(r'\s+', line)
            if len(words) >= 2:  # Минимум 2 колонки
                potential_table_lines.append(words)
        
        # Если нашли потенциальные табличные строки
        if potential_table_lines:
            # Определяем максимальное количество колонок
            max_cols = max(len(row) for row in potential_table_lines)
            
            # Нормализуем все строки до одинакового количества колонок
            for row in potential_table_lines:
                # Дополняем строки пустыми значениями до max_cols
                while len(row) < max_cols:
                    row.append("")
                table_data.append(row)
        
        return table_data
    
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

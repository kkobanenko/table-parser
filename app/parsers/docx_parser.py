import pandas as pd
from docx import Document
from typing import List, Dict
from utils.logger import setup_logger

logger = setup_logger('docx_parser')


class DOCXParser:
    """Парсер для извлечения таблиц из документов Word"""

    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Извлечение таблиц из DOCX файла

        Args:
            file_path (str): Путь к DOCX файлу

        Returns:
            List[Dict]: Список извлеченных таблиц
        """
        tables = []

        try:
            doc = Document(file_path)

            for table_idx, table in enumerate(doc.tables):
                try:
                    # Извлечение данных из таблицы
                    data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        data.append(row_data)

                    # Создание DataFrame
                    if len(data) > 1:
                        df = pd.DataFrame(data[1:], columns=data[0])

                        tables.append({
                            'data': df,
                            'sheet_name': f'DOCX_Table_{table_idx + 1}',
                            'source': 'docx_parser'
                        })

                except Exception as e:
                    logger.error(f"Ошибка обработки таблицы {table_idx}: {e}")

        except Exception as e:
            logger.error(f"Общая ошибка парсинга DOCX: {e}")

        return tables
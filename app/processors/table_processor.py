import pandas as pd
import numpy as np
from typing import Dict
from processors.column_mappings import ColumnMappings
from utils.logger import setup_logger

logger = setup_logger('table_processor')


class TableProcessor:
    """
    Процессор для трансформации и очистки таблиц
    Применяет стандартизацию и типизацию данных
    """

    def process_table(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Основной метод обработки таблицы

        Args:
            df (pd.DataFrame): Исходная таблица
            sheet_name (str): Имя листа

        Returns:
            Dict: Обработанная таблица с метаданными
        """
        try:
            # Очистка таблицы
            df = self._clean_dataframe(df)

            # Стандартизация заголовков
            df = self._normalize_headers(df)

            # Определение и конвертация типов
            df = self._convert_column_types(df)

            return {
                'dataframe': df,
                'sheet_name': sheet_name,
                'rows_count': len(df),
                'columns_count': len(df.columns)
            }

        except Exception as e:
            logger.error(f"Ошибка обработки таблицы: {e}")
            return {
                'dataframe': df,
                'sheet_name': sheet_name,
                'error': str(e)
            }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка таблицы от пустых и дублированных данных

        Args:
            df (pd.DataFrame): Исходная таблица

        Returns:
            pd.DataFrame: Очищенная таблица
        """
        # Удаление полностью пустых строк и столбцов
        df = df.dropna(how='all')

        # Удаление дубликатов
        df = df.drop_duplicates()

        # Очистка строковых столбцов
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        return df

    def _normalize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартизация заголовков таблицы

        Args:
            df (pd.DataFrame): Исходная таблица

        Returns:
            pd.DataFrame: Таблица с нормализованными заголовками
        """
        df.columns = [
            ColumnMappings.get_standard_column(col)
            for col in df.columns
        ]
        return df

    def _convert_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Определение и конвертация типов данных

        Args:
            df (pd.DataFrame): Исходная таблица

        Returns:
            pd.DataFrame: Таблица с корректными типами
        """
        for col in df.columns:
            # Попытка конвертации в числовой тип
            numeric_series = pd.to_numeric(
                df[col].apply(
                    lambda x: str(x).replace(' ', '')
                    .replace(',', '.')
                    .replace('%', '')
                ),
                errors='coerce'
            )

            # Конвертация, если более 50% значений успешно преобразованы
            if numeric_series.notna().mean() > 0.5:
                df[col] = numeric_series
                continue

            # Распознавание даты
            date_series = pd.to_datetime(
                df[col],
                errors='coerce',
                format='mixed'
            )

            if date_series.notna().mean() > 0.5:
                df[col] = date_series

        return df
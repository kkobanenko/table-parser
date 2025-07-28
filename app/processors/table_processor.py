import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, List
import json
from utils.logger import setup_logger

logger = setup_logger('table_processor')


class ColumnMappings:
    """Расширенный класс для маппинга колонок с учетом специфики медицинских прайс-листов"""

    HEADER_REPLACEMENTS = {
        # Наименование
        'наименование': 'name',
        'торговое наименование': 'name',
        'наименование препарата': 'name',
        'название': 'name',
        'мнн, дозировка, фасовка': 'name',

        # Синонимы для полного наименования
        'полное наименование': 'full_name',
        'наименование препарата с дозировкой': 'full_name',
        'международное непатентованное наименование': 'full_name',

        # Цена
        'цена': 'price',
        'цена прайса с ндс': 'price_with_vat',
        'цена, руб': 'price',
        'без НДС, руб.': 'price_without_vat',
        'с НДС, руб.': 'price_with_vat',
        'зарегистрированная предельная отпускная цена производителя (цена жнвлп)': 'regulated_price',
        'минимальная цена': 'min_price',
        'стоимость': 'price',

        # Дополнительные поля цены
        'оптовая цена': 'wholesale_price',
        'розничная цена': 'retail_price',

        # Количество
        'количество': 'quantity',
        'остаток': 'quantity',
        'доступное количество': 'quantity',

        # Идентификаторы
        'серия': 'series',
        'партия': 'batch',
        'артикул': 'article',
        'код': 'code',
        'код номенклатуры': 'nomenclature_code',

        # Дополнительные классификаторы
        'категория': 'category',
        'группа товаров': 'product_group',

        # Производитель
        'производитель': 'manufacturer',
        'страна производителя': 'manufacturer_country',

        # Дополнительные характеристики
        'форма выпуска': 'dosage_form',
        'дозировка': 'dosage',
        'фасовка': 'packaging',

        # Регистрационные данные
        'регистрационный номер': 'registration_number',
        'дата регистрации': 'registration_date'
    }

    PRICE_COLUMNS = {
        'price_with_vat': [
            'цена с ндс', 'с ндс, руб.',
            'цена прайса с ндс', 'цена розничная с ндс'
        ],
        'price_without_vat': [
            'цена без ндс', 'без ндс, руб.',
            'цена оптовая без ндс'
        ],
        'regulated_price': [
            'жнвлп', 'зарегистрированная цена',
            'предельная цена'
        ],
        'quantity': [
            'количество', 'остаток', 'доступное количество',
            'кол-во', 'remainder'
        ],
        'article': [
            'артикул', 'код', 'code', 'product_id'
        ]
    }

    @classmethod
    def get_standard_column(cls, column_name: str) -> str:
        """Преобразование колонки к стандартному виду"""
        clean_name = column_name.lower().strip()
        clean_name = re.sub(r'[^a-zа-я0-9\s]', '', clean_name)

        for key, value in cls.HEADER_REPLACEMENTS.items():
            if key in clean_name:
                return value

        return clean_name

    @classmethod
    def match_column_type(cls, column_name: str) -> str:
        """Определение типа колонки"""
        column_name = column_name.lower()

        type_matching = {
            'price': ['цена', 'стоимость', 'price', 'cost'],
            'quantity': ['количество', 'остаток', 'amount', 'quantity'],
            'article': ['артикул', 'код', 'code', 'id'],
            'series': ['серия', 'партия', 'batch', 'lot'],
            'name': ['наименование', 'название', 'name', 'title']
        }

        for col_type, keywords in type_matching.items():
            for keyword in keywords:
                if keyword in column_name:
                    return col_type

        return 'unknown'


class TableProcessor:
    """Обработчик и форматировщик таблиц"""

    def process_table(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Основной метод обработки таблицы"""
        try:
            # Первичная очистка
            df = self._clean_dataframe(df)

            # Стандартизация колонок
            df = self._normalize_columns(df)

            # Определение типов данных
            df = self._detect_and_convert_types(df)

            return {
                'dataframe': df,
                'sheet_name': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'processing_details': {
                    'timestamp': datetime.now().isoformat(),
                    'sheet_name': sheet_name
                }
            }

        except Exception as e:
            logger.error(f"Ошибка обработки таблицы: {e}")
            return {
                'dataframe': df,
                'sheet_name': sheet_name,
                'error': str(e)
            }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Первичная очистка DataFrame"""
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        df = df.drop_duplicates()

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Стандартизация названий колонок"""
        df.columns = [
            ColumnMappings.get_standard_column(col)
            for col in df.columns
        ]
        return df

    def _detect_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определение и конвертация типов данных"""
        for col in df.columns:
            # Попытка распознать числовые значения
            numeric_series = pd.to_numeric(
                df[col].apply(
                    lambda x: str(x).replace(' ', '')
                    .replace(',', '.')
                    .replace('%', '')
                ),
                errors='coerce'
            )

            # Если более 50% значений сконвертировались
            if numeric_series.notna().mean() > 0.5:
                df[col] = numeric_series
                continue

            # Распознавание дат
            date_formats = [
                '%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y',
                '%m/%d/%Y', '%Y.%m.%d', '%d-%m-%Y'
            ]

            for date_format in date_formats:
                dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                if dates.notna().mean() > 0.5:
                    df[col] = dates
                    break

        return df
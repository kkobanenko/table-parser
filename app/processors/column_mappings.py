import re
from typing import Dict


class ColumnMappings:
    """
    Класс для стандартизации и маппинга названий колонок
    """

    HEADER_REPLACEMENTS = {
        # Наименование
        'наименование': 'name',
        'наименование товара': 'name',
        'торговое наименование': 'name',
        'наименование препарата': 'name',
        'название': 'name',
        'мнн, дозировка, фасовка': 'name',

        # Синонимы для полного наименования
        'полное наименование': 'full_name',
        'наименование препарата с дозировкой': 'full_name',
        'международное непатентованное наименование': 'full_name',
        'мнн': 'mnn',

        # Цена
        'цена': 'price',
        'цена прайса с ндс': 'price_with_vat',
        'цена, руб': 'price',
        'без НДС, руб.': 'price_without_vat',
        'с НДС, руб.': 'price_with_vat',
        'зарегистрированная предельная отпускная цена производителя (цена жнвлп)': 'regulated_price',
        'минимальная цена': 'min_price',
        'стоимость': 'price',
        'ндс': 'vat',

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
        """
        Стандартизация названия колонки

        Args:
            column_name (str): Исходное название

        Returns:
            str: Стандартизированное название
        """
        # Приведение к нижнему регистру и очистка
        clean_name = str(column_name).lower().strip()
        clean_name = re.sub(r'[^a-zа-я0-9\s]', '', clean_name)

        # Поиск соответствия в словаре
        for key, value in cls.HEADER_REPLACEMENTS.items():
            if key in clean_name:
                return value

        return clean_name
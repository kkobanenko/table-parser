import re
from typing import Dict


class ColumnMappings:
    """
    Класс для стандартизации и маппинга названий колонок
    """

    HEADER_REPLACEMENTS: Dict[str, str] = {
        'наименование': 'name',
        'название': 'name',
        'цена': 'price',
        'стоимость': 'price',
        'количество': 'quantity',
        'серия': 'series',
        'артикул': 'article',
        'код': 'code'
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
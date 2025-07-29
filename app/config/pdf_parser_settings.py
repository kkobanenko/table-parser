import os
import json
from typing import Dict, Any
import streamlit as st


class PDFParserSettings:
    """
    Управление настройками PDF парсера
    Поддерживает сохранение и загрузку пользовательских настроек
    """

    DEFAULT_SETTINGS: Dict[str, Any] = {
        "line_margin": 0.5,
        "char_margin": 2.0,
        "line_overlap": 0.5,
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "rotation_angles": [0, 90, 180, 270],
        "pages": "all",
        "guess": True
    }

    SETTINGS_PATH: str = os.path.join(
        os.path.dirname(__file__),
        'pdf_parser_config.json'
    )

    @classmethod
    def load_settings(cls) -> Dict[str, Any]:
        """
        Загрузка настроек парсера PDF

        Returns:
            Dict[str, Any]: Словарь настроек
        """
        try:
            if os.path.exists(cls.SETTINGS_PATH):
                with open(cls.SETTINGS_PATH, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                    # Слияние пользовательских и дефолтных настроек
                    return {**cls.DEFAULT_SETTINGS, **user_settings}
            return cls.DEFAULT_SETTINGS
        except Exception as e:
            st.error(f"Ошибка загрузки настроек PDF парсера: {e}")
            return cls.DEFAULT_SETTINGS

    @classmethod
    def save_settings(cls, settings: Dict[str, Any]):
        """
        Сохранение пользовательских настроек

        Args:
            settings (Dict[str, Any]): Словарь настроек для сохранения
        """
        try:
            with open(cls.SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            st.success("Настройки PDF парсера сохранены!")
        except Exception as e:
            st.error(f"Ошибка сохранения настроек: {e}")

    @classmethod
    def get_streamlit_settings_ui(cls) -> Dict[str, Any]:
        """
        Создание UI настроек PDF парсера в Streamlit

        Returns:
            Dict[str, Any]: Выбранные пользователем настройки
        """
        # Загрузка текущих настроек
        settings = cls.load_settings()

        # Создание UI для настроек
        with st.form("pdf_parser_settings"):
            st.subheader("🔧 Настройки PDF парсера")

            col1, col2 = st.columns(2)

            with col1:
                settings['line_margin'] = st.slider(
                    "Margin линий",
                    min_value=0.1,
                    max_value=5.0,
                    value=settings['line_margin'],
                    step=0.1,
                    help="Точность определения горизонтальных линий"
                )

                settings['vertical_strategy'] = st.selectbox(
                    "Стратегия вертикальных границ",
                    ['lines', 'explicit', 'implied'],
                    index=['lines', 'explicit', 'implied'].index(settings['vertical_strategy'])
                )

            with col2:
                settings['char_margin'] = st.slider(
                    "Margin символов",
                    min_value=0.1,
                    max_value=5.0,
                    value=settings['char_margin'],
                    step=0.1,
                    help="Точность определения вертикальных линий"
                )

                settings['horizontal_strategy'] = st.selectbox(
                    "Стратегия горизонтальных границ",
                    ['lines', 'explicit', 'implied'],
                    index=['lines', 'explicit', 'implied'].index(settings['horizontal_strategy'])
                )

            settings['line_overlap'] = st.slider(
                "Перекрытие линий",
                min_value=0.0,
                max_value=1.0,
                value=settings['line_overlap'],
                step=0.1,
                help="Степень перекрытия линий при детекции таблиц"
            )

            settings['rotation_angles'] = st.multiselect(
                "Углы поворота для исследования",
                [0, 90, 180, 270],
                default=settings['rotation_angles'],
                help="Углы поворота документа для поиска таблиц"
            )

            # Кнопка сохранения
            submitted = st.form_submit_button("💾 Сохранить настройки")

            if submitted:
                cls.save_settings(settings)

        return settings
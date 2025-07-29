import os
import json
from typing import Dict, Any
from pathlib import Path
import tempfile
import streamlit as st


class PDFParserSettings:
    """
    Управление настройками PDF‑парсера.
    Поддерживает сохранение и загрузку пользовательских настроек.
    """

    # 1) каталог для временных PNG‑страниц (OCR)
    CACHE_DIR: Path = Path(tempfile.gettempdir()) / "pdfparser_cache"
    CACHE_DIR.mkdir(exist_ok=True)

    # 2) дефолтные значения
    DEFAULT_SETTINGS: Dict[str, Any] = {
        # pdfplumber / tabula
        "line_margin": 0.5,
        "char_margin": 2.0,
        "line_overlap": 0.5,
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "rotation_angles": [0, 90, 180, 270],
        "pages": "all",
        "guess": True,
        # Camelot
        "camelot_flavor": "stream",
        # OCR fallback
        "dpi": 300,
        "ocr": {
            "rotate_auto": True,
            "enhance_contrast": True,
            "denoise": True,
            "psm": 6,
        },
        # сервис
        "cache_dir": str(CACHE_DIR),
    }

    SETTINGS_PATH: str = os.path.join(
        os.path.dirname(__file__),
        "pdf_parser_config.json",
    )

    # ------------------------------------------------------------------ #
    # --- (де)сериализация --------------------------------------------- #

    @classmethod
    def load_settings(cls) -> Dict[str, Any]:
        """Загрузка настроек, с подмешиванием дефолтов."""
        try:
            if os.path.exists(cls.SETTINGS_PATH):
                with open(cls.SETTINGS_PATH, "r", encoding="utf-8") as f:
                    user_settings = json.load(f) or {}
                return {**cls.DEFAULT_SETTINGS, **user_settings}
            return cls.DEFAULT_SETTINGS
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка загрузки настроек PDF‑парсера: {exc}")
            return cls.DEFAULT_SETTINGS

    @classmethod
    def save_settings(cls, settings: Dict[str, Any]) -> None:
        """Сохранение пользовательских настроек."""
        try:
            with open(cls.SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            st.success("Настройки PDF‑парсера сохранены!")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка сохранения настроек: {exc}")

    # ------------------------------------------------------------------ #
    # --- Streamlit UI -------------------------------------------------- #

    @classmethod
    def get_streamlit_settings_ui(cls) -> Dict[str, Any]:
        """Форма настройки парсера (без вложенных expanders)."""
        settings = cls.load_settings()

        with st.form("pdf_parser_settings"):
            st.subheader("🔧 Настройки PDF‑парсера")

            # базовые параметры pdfplumber
            col1, col2 = st.columns(2)
            with col1:
                settings["line_margin"] = st.slider(
                    "line_margin",
                    0.1, 5.0, settings["line_margin"], 0.1,
                )
                settings["vertical_strategy"] = st.selectbox(
                    "vertical_strategy",
                    ["lines", "explicit", "implied"],
                    index=["lines", "explicit", "implied"].index(
                        settings["vertical_strategy"]
                    ),
                )
            with col2:
                settings["char_margin"] = st.slider(
                    "char_margin",
                    0.1, 5.0, settings["char_margin"], 0.1,
                )
                settings["horizontal_strategy"] = st.selectbox(
                    "horizontal_strategy",
                    ["lines", "explicit", "implied"],
                    index=["lines", "explicit", "implied"].index(
                        settings["horizontal_strategy"]
                    ),
                )

            # Camelot flavour
            settings["camelot_flavor"] = st.selectbox(
                "Camelot flavour",
                ["stream", "lattice"],
                index=["stream", "lattice"].index(settings["camelot_flavor"]),
            )

            # DPI для OCR
            settings["dpi"] = st.slider(
                "DPI для pdf2image (OCR‑fallback)",
                150, 600, settings["dpi"], 25,
            )

            # --- OCR‑настройки ----------------------------------------- #
            show_ocr = st.checkbox(
                "Показать дополнительные настройки OCR",
                value=False,
            )
            if show_ocr:
                settings["ocr"]["rotate_auto"] = st.checkbox(
                    "Авто‑поворот",
                    value=settings["ocr"]["rotate_auto"],
                )
                settings["ocr"]["enhance_contrast"] = st.checkbox(
                    "Повысить контраст",
                    value=settings["ocr"]["enhance_contrast"],
                )
                settings["ocr"]["denoise"] = st.checkbox(
                    "Денойзинг",
                    value=settings["ocr"]["denoise"],
                )
                settings["ocr"]["psm"] = st.slider(
                    "Tesseract PSM",
                    3, 13, settings["ocr"]["psm"],
                )
            # ----------------------------------------------------------- #

            if st.form_submit_button("💾 Сохранить"):
                cls.save_settings(settings)

        return settings

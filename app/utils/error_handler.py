from typing import Dict, Any, Optional
import traceback
from utils.logger import setup_logger

logger = setup_logger('error_handler')


class ErrorHandler:
    """
    Централизованная обработка ошибок в приложении
    """

    @staticmethod
    def handle_error(
            error: Exception,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Стандартизированная обработка ошибок

        Args:
            error (Exception): Перехваченное исключение
            context (Optional[Dict]): Дополнительный контекст ошибки

        Returns:
            Dict[str, Any]: Структурированное описание ошибки
        """
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }

        if context:
            error_details['context'] = context

        # Логирование ошибки
        logger.error(
            f"Ошибка: {error_details['error_type']} - {error_details['error_message']}\n"
            f"Трассировка: {error_details['traceback']}"
        )

        return error_details
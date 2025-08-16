#!/usr/bin/env python
# -*- python -*-

import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


def handle_exceptions(
    exception_handlers: Optional[
        Dict[Type[Exception], Callable[[Exception], Any]]
    ] = None,
    *additional_handlers: Dict[Type[Exception], Callable[[Exception], Any]],
    fallback_handler: Optional[Callable[[Exception], Any]] = None,
    log_level: str = "error",
    log_traceback: bool = True,
    reraise_on_fallback: bool = False,
) -> Callable:
    """
    Decorator for handling exceptions with custom mapping and fallback handlers.

    This decorator wraps a function to provide structured exception handling with
    customizable exception type-to-handler mappings. It allows for specific handling
    of different exception types and provides a fallback mechanism for uncaught exceptions.

    Args:
        exception_handlers: Primary dictionary mapping exception types to handler functions.
            Each handler function should accept the exception as an argument and return a value.
        *additional_handlers: Additional handler dictionaries to merge with the primary one.
            Later dictionaries will override earlier ones for conflicting exception types.
        fallback_handler: Optional function to handle any uncaught exceptions.
            If not provided, exceptions will be reraised or a generic error will be raised.
        log_level: Logging level for unhandled exceptions. Options: 'debug', 'info', 'warning', 'error', 'critical'.
            Defaults to 'error'.
        log_traceback: Whether to log the full traceback (True) or just the exception message (False).
            Defaults to True.
        reraise_on_fallback: If True, reraise the original exception when fallback is used.
            If False, wrap in a generic Exception. Defaults to False.

    Returns:
        Callable: The decorated function with exception handling.

    Raises:
        ValueError: If an invalid log_level is provided.
        Exception: If an uncaught exception occurs and no fallback_handler is provided.

    Example:
        ```python
        # Define exception handlers
        db_handlers = {
            ConnectionError: lambda e: {"success": False, "error": "Database unavailable"},
            IntegrityError: lambda e: {"success": False, "error": "Data integrity violation"}
        }

        validation_handlers = {
            ValueError: lambda e: {"success": False, "error": f"Invalid value: {str(e)}"},
            TypeError: lambda e: {"success": False, "error": f"Type error: {str(e)}"}
        }

        # Apply the decorator
        @handle_exceptions(
            db_handlers,
            validation_handlers,
            fallback_handler=create_error_response,
            log_level="warning",
            log_traceback=True
        )
        def create_user(user_data):
            # Function implementation that might raise exceptions
            pass
        ```
    """
    # Validate log_level
    valid_log_levels = {"debug", "info", "warning", "error", "critical"}
    if log_level.lower() not in valid_log_levels:
        raise ValueError(
            f"Invalid log_level '{log_level}'. Must be one of: {valid_log_levels}"
        )

    # Merge all exception handlers
    merged_handlers = exception_handlers.copy() if exception_handlers else {}
    for handler_dict in additional_handlers:
        if not isinstance(handler_dict, dict):
            raise TypeError("All additional_handlers must be dictionaries")
        merged_handlers.update(handler_dict)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_type = type(e)

                # Try to find a specific handler
                for exc_type, handler in merged_handlers.items():
                    if isinstance(e, exc_type):
                        logger.debug(
                            "Handling %s in %s with specific handler",
                            exception_type.__name__,
                            func.__name__,
                        )
                        try:
                            return handler(e)
                        except Exception as handler_error:
                            logger.error(
                                "Exception handler for %s failed: %s",
                                exception_type.__name__,
                                handler_error,
                            )
                            # Fall through to fallback handling
                            break

                # Log the unhandled exception
                log_func = getattr(logger, log_level.lower())
                if log_traceback:
                    log_func(
                        "Unhandled exception in %s:\n%s",
                        func.__name__,
                        traceback.format_exc(),
                    )
                else:
                    log_func(
                        "Unhandled exception in %s: %s (%s)",
                        func.__name__,
                        str(e),
                        exception_type.__name__,
                    )

                # Use fallback handler or reraise
                if fallback_handler:
                    try:
                        return fallback_handler(e)
                    except Exception as fallback_error:
                        logger.error("Fallback handler failed: %s", fallback_error)
                        if reraise_on_fallback:
                            raise e from fallback_error
                        else:
                            raise Exception(
                                f"Fallback handler failed: {fallback_error}"
                            ) from e
                else:
                    if reraise_on_fallback:
                        raise
                    else:
                        raise Exception(
                            f"Unhandled {exception_type.__name__}: {str(e)}"
                        ) from e

        return wrapper

    return decorator


def create_error_response(
    exception: Exception,
    include_traceback: bool = False,
    error_code: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Creates a standardized error response dictionary from an exception.

    This function provides a consistent error response format that's particularly
    useful for APIs and service interfaces.

    Args:
        exception: The exception that was caught.
        include_traceback: Whether to include the full traceback in the response.
            Defaults to False for security reasons.
        error_code: Custom error code to include. If None, uses 9999 as default.

    Returns:
        Dict[str, Any]: A dictionary containing standardized error information with keys:
            - success: Always False, indicating failure
            - error: The exception message or a generic message
            - error_code: Error code (custom or 9999 for unhandled exceptions)
            - error_type: The name of the exception class
            - traceback: Full traceback (only if include_traceback=True)

    Example:
        ```python
        try:
            result = risky_operation()
        except ValueError as e:
            return create_error_response(e, error_code=400)
        ```
    """
    response = {
        "success": False,
        "error": str(exception) or "An unexpected error occurred",
        "error_code": error_code or 9999,
        "error_type": type(exception).__name__,
    }

    if include_traceback:
        response["traceback"] = traceback.format_exc()

    return response


# Convenience function for common use cases
def handle_common_exceptions(
    include_value_error: bool = True,
    include_type_error: bool = True,
    include_key_error: bool = True,
    include_attribute_error: bool = True,
    custom_handlers: Optional[Dict[Type[Exception], Callable[[Exception], Any]]] = None,
    **decorator_kwargs,
) -> Callable:
    """
    Convenience decorator that handles common Python exceptions with sensible defaults.

    Args:
        include_value_error: Whether to handle ValueError exceptions.
        include_type_error: Whether to handle TypeError exceptions.
        include_key_error: Whether to handle KeyError exceptions.
        include_attribute_error: Whether to handle AttributeError exceptions.
        custom_handlers: Additional custom exception handlers.
        **decorator_kwargs: Additional keyword arguments passed to handle_exceptions.

    Returns:
        Callable: The decorated function with common exception handling.

    Example:
        ```python
        @handle_common_exceptions(
            include_key_error=False,
            custom_handlers={ConnectionError: lambda e: {"error": "Connection failed"}},
            log_level="warning"
        )
        def process_data(data):
            # Function that might raise common exceptions
            pass
        ```
    """
    common_handlers = {}

    if include_value_error:
        common_handlers[ValueError] = lambda e: create_error_response(e, error_code=400)

    if include_type_error:
        common_handlers[TypeError] = lambda e: create_error_response(e, error_code=400)

    if include_key_error:
        common_handlers[KeyError] = lambda e: create_error_response(e, error_code=404)

    if include_attribute_error:
        common_handlers[AttributeError] = lambda e: create_error_response(
            e, error_code=500
        )

    if custom_handlers:
        common_handlers.update(custom_handlers)

    return handle_exceptions(
        common_handlers,
        fallback_handler=decorator_kwargs.pop(
            "fallback_handler", create_error_response
        ),
        **decorator_kwargs,
    )

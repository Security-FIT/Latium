"""
llms_utils
==========

Utility and handler submodule for LLM framework. Provides model loading, registry, and handler classes for extensible LLM support.

:copyright: 2025 Jakub Res
:license: MIT
"""

import importlib
import pkgutil

# Dynamically import all modules in this package with a _handler suffix
__all__ = []

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.endswith("_handler"):
        module = importlib.import_module(f".{module_name}", __name__)
        # Optionally, extend __all__ with the module's __all__ if it exists
        if hasattr(module, "__all__"):
            __all__.extend(module.__all__)
        else:
            __all__.append(module_name)

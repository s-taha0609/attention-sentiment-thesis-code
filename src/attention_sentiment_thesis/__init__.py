"""Provider-neutral implementation of the final thesis specification.

Importing this package performs no file I/O, directory creation, or network access.
"""

from .spec import FINAL_SPEC, MODEL_PREDICTORS

__all__ = ["FINAL_SPEC", "MODEL_PREDICTORS"]
__version__ = "0.1.0"


"""Backward-compatible entry point for `from dataset import ...` imports."""

import importlib.util
import os

_PACKAGE_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
_PACKAGE_INIT = os.path.join(_PACKAGE_DIR, '__init__.py')

_spec = importlib.util.spec_from_file_location(
    '_dataset_package',
    _PACKAGE_INIT,
    submodule_search_locations=[_PACKAGE_DIR],
)
if _spec is None or _spec.loader is None:
    raise ImportError('Failed to load dataset package implementation.')

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if _name.startswith('__'):
        continue
    globals()[_name] = getattr(_module, _name)

_module_all = getattr(_module, '__all__', None)
if _module_all is None:
    __all__ = [name for name in globals() if not name.startswith('_')]
else:
    __all__ = list(_module_all)

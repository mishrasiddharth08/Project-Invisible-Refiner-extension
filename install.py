import sys
from pathlib import Path

required = '3.10'
try:
    if sys.version_info < (3, 10):
        raise RuntimeError(f'Python {required}+ is required')
except AttributeError:
    pass

print('PROJECT INVISIBLE — Universal Refiner: no extra dependencies required.')
print('It reuses the checkpoint already loaded by Forge; nothing else is installed.')

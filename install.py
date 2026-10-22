import sys

if sys.version_info < (3, 10):
    raise RuntimeError('Python 3.10+ is required')

print('PROJECT INVISIBLE - Refiner: no automatic dependency installation.')
print('Optional DLSS uses a separately supplied Visual Enhancer runtime; see docs/DLSS5.md.')

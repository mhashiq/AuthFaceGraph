import sys
import types

# Dummy lzma module mock to prevent ModuleNotFoundError: No module named '_lzma' on macOS environments
try:
    import lzma
except ImportError:
    dummy_lzma = types.ModuleType('lzma')
    dummy_lzma.LZMAError = Exception
    dummy_lzma.LZMACompressor = None
    dummy_lzma.LZMADecompressor = None
    dummy_lzma.open = None
    dummy_lzma.is_check_supported = lambda check: False
    sys.modules['lzma'] = dummy_lzma

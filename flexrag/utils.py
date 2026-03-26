import sys

def is_debug():
    return sys.gettrace() is not None
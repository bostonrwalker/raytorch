"""
Useful math functions
"""


def clamp(x, a, b):
    """
    Restrict a value x to range [a, b]
    """
    return min(b, max(a, x))

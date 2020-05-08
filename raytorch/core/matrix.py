import math
from raytorch.core.vector import Vector


class Matrix:
    
    """
    A 4 x 4 matrix
    """
    def __init__(self, a: (int, float, complex) = None, b: (int, float, complex) = None,
                 c: (int, float, complex) = None, d: (int, float, complex) = None, e: (int, float, complex) = None,
                 f: (int, float, complex) = None, g: (int, float, complex) = None, h: (int, float, complex) = None,
                 i: (int, float, complex) = None, j: (int, float, complex) = None, k: (int, float, complex) = None,
                 l: (int, float, complex) = None, m: (int, float, complex) = None, n: (int, float, complex) = None,
                 o: (int, float, complex) = None, p: (int, float, complex) = None):
        self.a = a if a is not None else 0.
        self.b = b if b is not None else 0.
        self.c = c if c is not None else 0.
        self.d = d if d is not None else 0.
        self.e = e if e is not None else 0.
        self.f = f if f is not None else 0.
        self.g = g if g is not None else 0.
        self.h = h if h is not None else 0.
        self.i = i if i is not None else 0.
        self.j = j if j is not None else 0.
        self.k = k if k is not None else 0.
        self.l = l if l is not None else 0.
        self.m = m if m is not None else 0.
        self.n = n if n is not None else 0.
        self.o = o if o is not None else 0.
        self.p = p if p is not None else 0.

    """
    Unary operators
    """

    def __pos__(self):
        """
        + operator
        """
        return Matrix(self.a, self.b, self.c, self.d,
                      self.e, self.f, self.g, self.h,
                      self.i, self.j, self.k, self.l,
                      self.m, self.n, self.o, self.p)

    def __neg__(self):
        """
        - operator
        """
        return Matrix(-self.a, -self.b, -self.c, -self.d,
                      -self.e, -self.f, -self.g, -self.h,
                      -self.i, -self.j, -self.k, -self.l,
                      -self.m, -self.n, -self.o, -self.p)

    def __invert__(self):
        """
        ~ operator
        """
        return NotImplemented  # TODO: implement matrix inversion

    def __abs__(self):
        """
        abs()
        """
        return Matrix(abs(self.a), abs(self.b), abs(self.c), abs(self.d),
                      abs(self.e), abs(self.f), abs(self.g), abs(self.h),
                      abs(self.i), abs(self.j), abs(self.k), abs(self.l),
                      abs(self.m), abs(self.n), abs(self.o), abs(self.p))

    def __floor__(self):
        """
        math.floor()
        """
        return Matrix(math.floor(self.a), math.floor(self.b), math.floor(self.c), math.floor(self.d),
                      math.floor(self.e), math.floor(self.f), math.floor(self.g), math.floor(self.h),
                      math.floor(self.i), math.floor(self.j), math.floor(self.k), math.floor(self.l),
                      math.floor(self.m), math.floor(self.n), math.floor(self.o), math.floor(self.p))

    def __ceil__(self):
        """
        math.ceil()
        """
        return Matrix(math.ceil(self.a), math.ceil(self.b), math.ceil(self.c), math.ceil(self.d),
                      math.ceil(self.e), math.ceil(self.f), math.ceil(self.g), math.ceil(self.h),
                      math.ceil(self.i), math.ceil(self.j), math.ceil(self.k), math.ceil(self.l),
                      math.ceil(self.m), math.ceil(self.n), math.ceil(self.o), math.ceil(self.p))

    def __round__(self, n=None):
        """
        round()
        """
        return Matrix(round(self.a, n), round(self.b, n), round(self.c, n), round(self.d, n),
                      round(self.e, n), round(self.f, n), round(self.g, n), round(self.h, n),
                      round(self.i, n), round(self.j, n), round(self.k, n), round(self.l, n),
                      round(self.m, n), round(self.n, n), round(self.o, n), round(self.p, n))

    def __trunc__(self):
        """
        math.trunc()
        """
        return Matrix(math.trunc(self.a), math.trunc(self.b), math.trunc(self.c), math.trunc(self.d),
                      math.trunc(self.e), math.trunc(self.f), math.trunc(self.g), math.trunc(self.h),
                      math.trunc(self.i), math.trunc(self.j), math.trunc(self.k), math.trunc(self.l),
                      math.trunc(self.m), math.trunc(self.n), math.trunc(self.o), math.trunc(self.p))
    
    """
    Binary arithmetical operators
    """

    def __add__(self, other):
        """
        + operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(self.a + other, self.b + other, self.c + other, self.d + other,
                          self.e + other, self.f + other, self.g + other, self.h + other,
                          self.i + other, self.j + other, self.k + other, self.l + other,
                          self.m + other, self.n + other, self.o + other, self.p + other)
        elif isinstance(other, Vector):
            return Matrix(self.a + other.x, self.b + other.x, self.c + other.x, self.d + other.x,
                          self.e + other.y, self.f + other.y, self.g + other.y, self.h + other.y,
                          self.i + other.z, self.j + other.z, self.k + other.z, self.l + other.z,
                          self.m + other.w, self.n + other.w, self.o + other.w, self.p + other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                          self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h,
                          self.i + other.i, self.j + other.j, self.k + other.k, self.l + other.l,
                          self.m + other.m, self.n + other.n, self.o + other.o, self.p + other.p)
        else:
            return NotImplemented

    def __radd__(self, other):
        """
        + operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(other + self.a, other + self.b, other + self.c, other + self.d,
                          other + self.e, other + self.f, other + self.g, other + self.h,
                          other + self.i, other + self.j, other + self.k, other + self.l,
                          other + self.m, other + self.n, other + self.o, other + self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x + self.a, other.x + self.b, other.x + self.c, other.x + self.d,
                          other.y + self.e, other.y + self.f, other.y + self.g, other.y + self.h,
                          other.z + self.i, other.z + self.j, other.z + self.k, other.z + self.l,
                          other.w + self.m, other.w + self.n, other.w + self.o, other.w + self.p)
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        - operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(self.a - other, self.b - other, self.c - other, self.d - other,
                          self.e - other, self.f - other, self.g - other, self.h - other,
                          self.i - other, self.j - other, self.k - other, self.l - other,
                          self.m - other, self.n - other, self.o - other, self.p - other)
        elif isinstance(other, Vector):
            return Matrix(self.a - other.x, self.b - other.x, self.c - other.x, self.d - other.x,
                          self.e - other.y, self.f - other.y, self.g - other.y, self.h - other.y,
                          self.i - other.z, self.j - other.z, self.k - other.z, self.l - other.z,
                          self.m - other.w, self.n - other.w, self.o - other.w, self.p - other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d,
                          self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h,
                          self.i - other.i, self.j - other.j, self.k - other.k, self.l - other.l,
                          self.m - other.m, self.n - other.n, self.o - other.o, self.p - other.p)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        - operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(other - self.a, other - self.b, other - self.c, other - self.d,
                          other - self.e, other - self.f, other - self.g, other - self.h,
                          other - self.i, other - self.j, other - self.k, other - self.l,
                          other - self.m, other - self.n, other - self.o, other - self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x - self.a, other.x - self.b, other.x - self.c, other.x - self.d,
                          other.y - self.e, other.y - self.f, other.y - self.g, other.y - self.h,
                          other.z - self.i, other.z - self.j, other.z - self.k, other.z - self.l,
                          other.w - self.m, other.w - self.n, other.w - self.o, other.w - self.p)
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        * operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(self.a * other, self.b * other, self.c * other, self.d * other,
                          self.e * other, self.f * other, self.g * other, self.h * other,
                          self.i * other, self.j * other, self.k * other, self.l * other,
                          self.m * other, self.n * other, self.o * other, self.p * other)
        elif isinstance(other, Vector):
            return Matrix(self.a * other.x, self.b * other.x, self.c * other.x, self.d * other.x,
                          self.e * other.y, self.f * other.y, self.g * other.y, self.h * other.y,
                          self.i * other.z, self.j * other.z, self.k * other.z, self.l * other.z,
                          self.m * other.w, self.n * other.w, self.o * other.w, self.p * other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a * other.a, self.b * other.b, self.c * other.c, self.d * other.d,
                          self.e * other.e, self.f * other.f, self.g * other.g, self.h * other.h,
                          self.i * other.i, self.j * other.j, self.k * other.k, self.l * other.l,
                          self.m * other.m, self.n * other.n, self.o * other.o, self.p * other.p)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        * operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(other * self.a, other * self.b, other * self.c, other * self.d,
                          other * self.e, other * self.f, other * self.g, other * self.h,
                          other * self.i, other * self.j, other * self.k, other * self.l,
                          other * self.m, other * self.n, other * self.o, other * self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x * self.a, other.x * self.b, other.x * self.c, other.x * self.d,
                          other.y * self.e, other.y * self.f, other.y * self.g, other.y * self.h,
                          other.z * self.i, other.z * self.j, other.z * self.k, other.z * self.l,
                          other.w * self.m, other.w * self.n, other.w * self.o, other.w * self.p)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """
        @ operator
        """
        if isinstance(other, Vector):
            return Vector(self.a * other.x + self.b * other.y + self.c * other.z + self.d * other.w,
                          self.e * other.x + self.f * other.y + self.g * other.z + self.h * other.w,
                          self.i * other.x + self.j * other.y + self.k * other.z + self.l * other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a * other.a + self.b * other.e + self.c * other.i + self.d * other.m,
                          self.a * other.b + self.b * other.f + self.c * other.j + self.d * other.n,
                          self.a * other.c + self.b * other.g + self.c * other.k + self.d * other.o,
                          self.a * other.d + self.b * other.h + self.c * other.l + self.d * other.p,
                          self.e * other.a + self.f * other.e + self.g * other.i + self.h * other.m,
                          self.e * other.b + self.f * other.f + self.g * other.j + self.h * other.n,
                          self.e * other.c + self.f * other.g + self.g * other.k + self.h * other.o,
                          self.e * other.d + self.f * other.h + self.g * other.l + self.h * other.p,
                          self.i * other.a + self.j * other.e + self.k * other.i + self.l * other.m,
                          self.i * other.b + self.j * other.f + self.k * other.j + self.l * other.n,
                          self.i * other.c + self.j * other.g + self.k * other.k + self.l * other.o,
                          self.i * other.d + self.j * other.h + self.k * other.l + self.l * other.p,
                          self.m * other.a + self.n * other.e + self.o * other.i + self.p * other.m,
                          self.m * other.b + self.n * other.f + self.o * other.j + self.p * other.n,
                          self.m * other.c + self.n * other.g + self.o * other.k + self.p * other.o,
                          self.m * other.d + self.n * other.h + self.o * other.l + self.p * other.p)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        @ operator
        """
        if isinstance(other, Vector):
            return Matrix(self.a * other.x + self.e * other.y + self.i * other.z + self.m * other.w,
                          self.b * other.x + self.f * other.y + self.j * other.z + self.n * other.w,
                          self.c * other.x + self.g * other.y + self.k * other.z + self.o * other.w,
                          self.d * other.x + self.h * other.y + self.l * other.z + self.p * other.w)
        else:
            return NotImplemented

    def __pow__(self, power: float, modulo=None):
        """
        ** operator
        """
        if isinstance(power, int) or isinstance(power, float) or isinstance(power, complex):
            return Matrix(self.a ** power, self.b ** power, self.c ** power, self.d ** power,
                          self.e ** power, self.f ** power, self.g ** power, self.h ** power,
                          self.i ** power, self.j ** power, self.k ** power, self.l ** power,
                          self.m ** power, self.n ** power, self.o ** power, self.p ** power)
        elif isinstance(power, Vector):
            return Matrix(self.a ** power.x, self.b ** power.x, self.c ** power.x, self.d ** power.x,
                          self.e ** power.y, self.f ** power.y, self.g ** power.y, self.h ** power.y,
                          self.i ** power.z, self.j ** power.z, self.k ** power.z, self.l ** power.z,
                          self.m ** power.w, self.n ** power.w, self.o ** power.w, self.p ** power.w)
        elif isinstance(power, Matrix):
            return Matrix(self.a ** power.a, self.b ** power.b, self.c ** power.c, self.d ** power.d,
                          self.e ** power.e, self.f ** power.f, self.g ** power.g, self.h ** power.h,
                          self.i ** power.i, self.j ** power.j, self.k ** power.k, self.l ** power.l,
                          self.m ** power.m, self.n ** power.n, self.o ** power.o, self.p ** power.p)
        else:
            return NotImplemented

    def __rpow__(self, other):
        """
        ** operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(other ** self.a, other ** self.b, other ** self.c, other ** self.d,
                          other ** self.e, other ** self.f, other ** self.g, other ** self.h,
                          other ** self.i, other ** self.j, other ** self.k, other ** self.l,
                          other ** self.m, other ** self.n, other ** self.o, other ** self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x ** self.a, other.x ** self.b, other.x ** self.c, other.x ** self.d,
                          other.y ** self.e, other.y ** self.f, other.y ** self.g, other.y ** self.h,
                          other.z ** self.i, other.z ** self.j, other.z ** self.k, other.z ** self.l,
                          other.w ** self.m, other.w ** self.n, other.w ** self.o, other.w ** self.p)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        / operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(self.a / other, self.b / other, self.c / other, self.d / other,
                          self.e / other, self.f / other, self.g / other, self.h / other,
                          self.i / other, self.j / other, self.k / other, self.l / other,
                          self.m / other, self.n / other, self.o / other, self.p / other)
        elif isinstance(other, Vector):
            return Matrix(self.a / other.x, self.b / other.x, self.c / other.x, self.d / other.x,
                          self.e / other.y, self.f / other.y, self.g / other.y, self.h / other.y,
                          self.i / other.z, self.j / other.z, self.k / other.z, self.l / other.z,
                          self.m / other.w, self.n / other.w, self.o / other.w, self.p / other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a / other.a, self.b / other.b, self.c / other.c, self.d / other.d,
                          self.e / other.e, self.f / other.f, self.g / other.g, self.h / other.h,
                          self.i / other.i, self.j / other.j, self.k / other.k, self.l / other.l,
                          self.m / other.m, self.n / other.n, self.o / other.o, self.p / other.p)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        / operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix(other / self.a, other / self.b, other / self.c, other / self.d,
                          other / self.e, other / self.f, other / self.g, other / self.h,
                          other / self.i, other / self.j, other / self.k, other / self.l,
                          other / self.m, other / self.n, other / self.o, other / self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x / self.a, other.x / self.b, other.x / self.c, other.x / self.d,
                          other.y / self.e, other.y / self.f, other.y / self.g, other.y / self.h,
                          other.z / self.i, other.z / self.j, other.z / self.k, other.z / self.l,
                          other.w / self.m, other.w / self.n, other.w / self.o, other.w / self.p)
        else:
            return NotImplemented

    def __floordiv__(self, other):
        """
        // operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(self.a // other, self.b // other, self.c // other, self.d // other,
                          self.e // other, self.f // other, self.g // other, self.h // other,
                          self.i // other, self.j // other, self.k // other, self.l // other,
                          self.m // other, self.n // other, self.o // other, self.p // other)
        elif isinstance(other, Vector):
            return Matrix(self.a // other.x, self.b // other.x, self.c // other.x, self.d // other.x,
                          self.e // other.y, self.f // other.y, self.g // other.y, self.h // other.y,
                          self.i // other.z, self.j // other.z, self.k // other.z, self.l // other.z,
                          self.m // other.w, self.n // other.w, self.o // other.w, self.p // other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a // other.a, self.b // other.b, self.c // other.c, self.d // other.d,
                          self.e // other.e, self.f // other.f, self.g // other.g, self.h // other.h,
                          self.i // other.i, self.j // other.j, self.k // other.k, self.l // other.l,
                          self.m // other.m, self.n // other.n, self.o // other.o, self.p // other.p)
        else:
            return NotImplemented

    def __rfloordiv__(self, other):
        """
        // operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(other // self.a, other // self.b, other // self.c, other // self.d,
                          other // self.e, other // self.f, other // self.g, other // self.h,
                          other // self.i, other // self.j, other // self.k, other // self.l,
                          other // self.m, other // self.n, other // self.o, other // self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x // self.a, other.x // self.b, other.x // self.c, other.x // self.d,
                          other.y // self.e, other.y // self.f, other.y // self.g, other.y // self.h,
                          other.z // self.i, other.z // self.j, other.z // self.k, other.z // self.l,
                          other.w // self.m, other.w // self.n, other.w // self.o, other.w // self.p)
        else:
            return NotImplemented

    def __mod__(self, other):
        """
        % operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(self.a % other, self.b % other, self.c % other, self.d % other,
                          self.e % other, self.f % other, self.g % other, self.h % other,
                          self.i % other, self.j % other, self.k % other, self.l % other,
                          self.m % other, self.n % other, self.o % other, self.p % other)
        elif isinstance(other, Vector):
            return Matrix(self.a % other.x, self.b % other.x, self.c % other.x, self.d % other.x,
                          self.e % other.y, self.f % other.y, self.g % other.y, self.h % other.y,
                          self.i % other.z, self.j % other.z, self.k % other.z, self.l % other.z,
                          self.m % other.w, self.n % other.w, self.o % other.w, self.p % other.w)
        elif isinstance(other, Matrix):
            return Matrix(self.a % other.a, self.b % other.b, self.c % other.c, self.d % other.d,
                          self.e % other.e, self.f % other.f, self.g % other.g, self.h % other.h,
                          self.i % other.i, self.j % other.j, self.k % other.k, self.l % other.l,
                          self.m % other.m, self.n % other.n, self.o % other.o, self.p % other.p)
        else:
            return NotImplemented

    def __rmod__(self, other):
        """
        % operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(other % self.a, other % self.b, other % self.c, other % self.d,
                          other % self.e, other % self.f, other % self.g, other % self.h,
                          other % self.i, other % self.j, other % self.k, other % self.l,
                          other % self.m, other % self.n, other % self.o, other % self.p)
        elif isinstance(other, Vector):
            return Matrix(other.x % self.a, other.x % self.b, other.x % self.c, other.x % self.d,
                          other.y % self.e, other.y % self.f, other.y % self.g, other.y % self.h,
                          other.z % self.i, other.z % self.j, other.z % self.k, other.z % self.l,
                          other.w % self.m, other.w % self.n, other.w % self.o, other.w % self.p)
        else:
            return NotImplemented

    def __divmod__(self, other):
        """
        divmod() function
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(divmod(self.a, other), divmod(self.b, other), divmod(self.c, other), divmod(self.d, other),
                          divmod(self.e, other), divmod(self.f, other), divmod(self.g, other), divmod(self.h, other),
                          divmod(self.i, other), divmod(self.j, other), divmod(self.k, other), divmod(self.l, other),
                          divmod(self.m, other), divmod(self.n, other), divmod(self.o, other), divmod(self.p, other))
        elif isinstance(other, Vector):
            return Matrix(divmod(self.a, other.x), divmod(self.b, other.x), divmod(self.c, other.x),
                          divmod(self.d, other.x), divmod(self.e, other.y), divmod(self.f, other.y),
                          divmod(self.g, other.y), divmod(self.h, other.y), divmod(self.i, other.z),
                          divmod(self.j, other.z), divmod(self.k, other.z), divmod(self.l, other.z),
                          divmod(self.m, other.w), divmod(self.n, other.w), divmod(self.o, other.w),
                          divmod(self.p, other.w))
        elif isinstance(other, Matrix):
            return Matrix(divmod(self.a, other.a), divmod(self.b, other.b), divmod(self.c, other.c),
                          divmod(self.d, other.d), divmod(self.e, other.e), divmod(self.f, other.f),
                          divmod(self.g, other.g), divmod(self.h, other.h), divmod(self.i, other.i),
                          divmod(self.j, other.j), divmod(self.k, other.k), divmod(self.l, other.l),
                          divmod(self.m, other.m), divmod(self.n, other.n), divmod(self.o, other.o),
                          divmod(self.p, other.p))
        else:
            return NotImplemented

    def __rdivmod__(self, other):
        """
        divmod() function
        """
        if isinstance(other, int) or isinstance(other, float):
            return Matrix(divmod(other, self.a), divmod(other, self.b), divmod(other, self.c), divmod(other, self.d),
                          divmod(other, self.e), divmod(other, self.f), divmod(other, self.g), divmod(other, self.h),
                          divmod(other, self.i), divmod(other, self.j), divmod(other, self.k), divmod(other, self.l),
                          divmod(other, self.m), divmod(other, self.n), divmod(other, self.o), divmod(other, self.p))
        elif isinstance(other, Vector):
            return Matrix(divmod(other.x, self.a), divmod(other.x, self.b), divmod(other.x, self.c),
                          divmod(other.x, self.d), divmod(other.y, self.e), divmod(other.y, self.f),
                          divmod(other.y, self.g), divmod(other.y, self.h), divmod(other.z, self.i),
                          divmod(other.z, self.j), divmod(other.z, self.k), divmod(other.z, self.l),
                          divmod(other.w, self.m), divmod(other.w, self.n), divmod(other.w, self.o),
                          divmod(other.w, self.p))
        else:
            return NotImplemented

    """
    Binary logical operators
    """

    def __eq__(self, other):
        """
        == operator
        """
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d and \
               self.e == other.e and self.f == other.f and self.g == other.g and self.h == other.h and \
               self.i == other.i and self.j == other.j and self.k == other.k and self.l == other.l and \
               self.m == other.m and self.n == other.n and self.o == other.o and self.p == other.p

    def __ne__(self, other):
        """
        != operator
        """
        return self.a != other.a and self.b != other.b and self.c != other.c and self.d != other.d and \
               self.e != other.e and self.f != other.f and self.g != other.g and self.h != other.h and \
               self.i != other.i and self.j != other.j and self.k != other.k and self.l != other.l and \
               self.m != other.m and self.n != other.n and self.o != other.o and self.p != other.p

    def __gt__(self, other):
        """
        > operator
        """
        return self.a > other.a and self.b > other.b and self.c > other.c and self.d > other.d and \
               self.e > other.e and self.f > other.f and self.g > other.g and self.h > other.h and \
               self.i > other.i and self.j > other.j and self.k > other.k and self.l > other.l and \
               self.m > other.m and self.n > other.n and self.o > other.o and self.p > other.p

    def __ge__(self, other):
        """
        >= operator
        """
        return self.a >= other.a and self.b >= other.b and self.c >= other.c and self.d >= other.d and \
               self.e >= other.e and self.f >= other.f and self.g >= other.g and self.h >= other.h and \
               self.i >= other.i and self.j >= other.j and self.k >= other.k and self.l >= other.l and \
               self.m >= other.m and self.n >= other.n and self.o >= other.o and self.p >= other.p

    def __lt__(self, other):
        """
        < operator
        """
        return self.a < other.a and self.b < other.b and self.c < other.c and self.d < other.d and \
               self.e < other.e and self.f < other.f and self.g < other.g and self.h < other.h and \
               self.i < other.i and self.j < other.j and self.k < other.k and self.l < other.l and \
               self.m < other.m and self.n < other.n and self.o < other.o and self.p < other.p

    def __le__(self, other):
        """
        <= operator
        """
        return self.a <= other.a and self.b <= other.b and self.c <= other.c and self.d <= other.d and \
               self.e <= other.e and self.f <= other.f and self.g <= other.g and self.h <= other.h and \
               self.i <= other.i and self.j <= other.j and self.k <= other.k and self.l <= other.l and \
               self.m <= other.m and self.n <= other.n and self.o <= other.o and self.p <= other.p

    """
    Inplace operators
    """

    def __iadd__(self, other):
        """
        += operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a += other
            self.b += other
            self.c += other
            self.d += other
            self.e += other
            self.f += other
            self.g += other
            self.h += other
            self.i += other
            self.j += other
            self.k += other
            self.l += other
            self.m += other
            self.n += other
            self.o += other
            self.p += other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a += other.x
            self.b += other.x
            self.c += other.x
            self.d += other.x
            self.e += other.y
            self.f += other.y
            self.g += other.y
            self.h += other.y
            self.i += other.z
            self.j += other.z
            self.k += other.z
            self.l += other.z
            self.m += other.w
            self.n += other.w
            self.o += other.w
            self.p += other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a += other.a
            self.b += other.b
            self.c += other.c
            self.d += other.d
            self.e += other.e
            self.f += other.f
            self.g += other.g
            self.h += other.h
            self.i += other.i
            self.j += other.j
            self.k += other.k
            self.l += other.l
            self.m += other.m
            self.n += other.n
            self.o += other.o
            self.p += other.p
        else:
            return NotImplemented
        return self

    def __isub__(self, other):
        """
        -= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a -= other
            self.b -= other
            self.c -= other
            self.d -= other
            self.e -= other
            self.f -= other
            self.g -= other
            self.h -= other
            self.i -= other
            self.j -= other
            self.k -= other
            self.l -= other
            self.m -= other
            self.n -= other
            self.o -= other
            self.p -= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a -= other.x
            self.b -= other.x
            self.c -= other.x
            self.d -= other.x
            self.e -= other.y
            self.f -= other.y
            self.g -= other.y
            self.h -= other.y
            self.i -= other.z
            self.j -= other.z
            self.k -= other.z
            self.l -= other.z
            self.m -= other.w
            self.n -= other.w
            self.o -= other.w
            self.p -= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a -= other.a
            self.b -= other.b
            self.c -= other.c
            self.d -= other.d
            self.e -= other.e
            self.f -= other.f
            self.g -= other.g
            self.h -= other.h
            self.i -= other.i
            self.j -= other.j
            self.k -= other.k
            self.l -= other.l
            self.m -= other.m
            self.n -= other.n
            self.o -= other.o
            self.p -= other.p
        else:
            return NotImplemented
        return self

    def __imul__(self, other):
        """
        *= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a *= other
            self.b *= other
            self.c *= other
            self.d *= other
            self.e *= other
            self.f *= other
            self.g *= other
            self.h *= other
            self.i *= other
            self.j *= other
            self.k *= other
            self.l *= other
            self.m *= other
            self.n *= other
            self.o *= other
            self.p *= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a *= other.x
            self.b *= other.x
            self.c *= other.x
            self.d *= other.x
            self.e *= other.y
            self.f *= other.y
            self.g *= other.y
            self.h *= other.y
            self.i *= other.z
            self.j *= other.z
            self.k *= other.z
            self.l *= other.z
            self.m *= other.w
            self.n *= other.w
            self.o *= other.w
            self.p *= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a *= other.a
            self.b *= other.b
            self.c *= other.c
            self.d *= other.d
            self.e *= other.e
            self.f *= other.f
            self.g *= other.g
            self.h *= other.h
            self.i *= other.i
            self.j *= other.j
            self.k *= other.k
            self.l *= other.l
            self.m *= other.m
            self.n *= other.n
            self.o *= other.o
            self.p *= other.p
        else:
            return NotImplemented
        return self

    def __imatmul__(self, other):
        """
        @= operator
        """
        if isinstance(other, Matrix):
            self.a, self.b, self.c, self.d, \
            self.e, self.f, self.g, self.h, \
            self.i, self.j, self.k, self.l, \
            self.m, self.n, self.o, self.p = (self @ other).astuple()
        else:
            return NotImplemented
        return self

    def __ipow__(self, other):
        """
        **= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a **= other
            self.b **= other
            self.c **= other
            self.d **= other
            self.e **= other
            self.f **= other
            self.g **= other
            self.h **= other
            self.i **= other
            self.j **= other
            self.k **= other
            self.l **= other
            self.m **= other
            self.n **= other
            self.o **= other
            self.p **= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a **= other.x
            self.b **= other.x
            self.c **= other.x
            self.d **= other.x
            self.e **= other.y
            self.f **= other.y
            self.g **= other.y
            self.h **= other.y
            self.i **= other.z
            self.j **= other.z
            self.k **= other.z
            self.l **= other.z
            self.m **= other.w
            self.n **= other.w
            self.o **= other.w
            self.p **= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a **= other.a
            self.b **= other.b
            self.c **= other.c
            self.d **= other.d
            self.e **= other.e
            self.f **= other.f
            self.g **= other.g
            self.h **= other.h
            self.i **= other.i
            self.j **= other.j
            self.k **= other.k
            self.l **= other.l
            self.m **= other.m
            self.n **= other.n
            self.o **= other.o
            self.p **= other.p
        else:
            return NotImplemented
        return self

    def __itruediv__(self, other):
        """
        /= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a /= other
            self.b /= other
            self.c /= other
            self.d /= other
            self.e /= other
            self.f /= other
            self.g /= other
            self.h /= other
            self.i /= other
            self.j /= other
            self.k /= other
            self.l /= other
            self.m /= other
            self.n /= other
            self.o /= other
            self.p /= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a /= other.x
            self.b /= other.x
            self.c /= other.x
            self.d /= other.x
            self.e /= other.y
            self.f /= other.y
            self.g /= other.y
            self.h /= other.y
            self.i /= other.z
            self.j /= other.z
            self.k /= other.z
            self.l /= other.z
            self.m /= other.w
            self.n /= other.w
            self.o /= other.w
            self.p /= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a /= other.a
            self.b /= other.b
            self.c /= other.c
            self.d /= other.d
            self.e /= other.e
            self.f /= other.f
            self.g /= other.g
            self.h /= other.h
            self.i /= other.i
            self.j /= other.j
            self.k /= other.k
            self.l /= other.l
            self.m /= other.m
            self.n /= other.n
            self.o /= other.o
            self.p /= other.p
        else:
            return NotImplemented
        return self

    def __ifloordiv__(self, other):
        """
        //= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a //= other
            self.b //= other
            self.c //= other
            self.d //= other
            self.e //= other
            self.f //= other
            self.g //= other
            self.h //= other
            self.i //= other
            self.j //= other
            self.k //= other
            self.l //= other
            self.m //= other
            self.n //= other
            self.o //= other
            self.p //= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a //= other.x
            self.b //= other.x
            self.c //= other.x
            self.d //= other.x
            self.e //= other.y
            self.f //= other.y
            self.g //= other.y
            self.h //= other.y
            self.i //= other.z
            self.j //= other.z
            self.k //= other.z
            self.l //= other.z
            self.m //= other.w
            self.n //= other.w
            self.o //= other.w
            self.p //= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a //= other.a
            self.b //= other.b
            self.c //= other.c
            self.d //= other.d
            self.e //= other.e
            self.f //= other.f
            self.g //= other.g
            self.h //= other.h
            self.i //= other.i
            self.j //= other.j
            self.k //= other.k
            self.l //= other.l
            self.m //= other.m
            self.n //= other.n
            self.o //= other.o
            self.p //= other.p
        else:
            return NotImplemented
        return self

    def __imod__(self, other):
        """
        %= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            """
            Apply scalar element-wise
            """
            self.a %= other
            self.b %= other
            self.c %= other
            self.d %= other
            self.e %= other
            self.f %= other
            self.g %= other
            self.h %= other
            self.i %= other
            self.j %= other
            self.k %= other
            self.l %= other
            self.m %= other
            self.n %= other
            self.o %= other
            self.p %= other
        if isinstance(other, Vector):
            """
            Broadcast Vector column-wise
            """
            self.a %= other.x
            self.b %= other.x
            self.c %= other.x
            self.d %= other.x
            self.e %= other.y
            self.f %= other.y
            self.g %= other.y
            self.h %= other.y
            self.i %= other.z
            self.j %= other.z
            self.k %= other.z
            self.l %= other.z
            self.m %= other.w
            self.n %= other.w
            self.o %= other.w
            self.p %= other.w
        elif isinstance(other, Matrix):
            """
            Element-wise
            """
            self.a %= other.a
            self.b %= other.b
            self.c %= other.c
            self.d %= other.d
            self.e %= other.e
            self.f %= other.f
            self.g %= other.g
            self.h %= other.h
            self.i %= other.i
            self.j %= other.j
            self.k %= other.k
            self.l %= other.l
            self.m %= other.m
            self.n %= other.n
            self.o %= other.o
            self.p %= other.p
        else:
            return NotImplemented
        return self

    """
    Other overloads
    """
    
    def __str__(self):
        return (f'Matrix{{[[{str(self.a)},{str(self.b)},{str(self.c)},{str(self.d)}],'
                f'[{str(self.e)},{str(self.f)},{str(self.g)},{str(self.h)}],'
                f'[{str(self.i)},{str(self.j)},{str(self.k)},{str(self.l)}],'
                f'[{str(self.m)},{str(self.n)},{str(self.o)},{str(self.p)}]]}}')
    
    def __repr__(self):
        return (f'Matrix{{[[{repr(self.a)},{repr(self.b)},{repr(self.c)},{repr(self.d)}],'
                f'[{repr(self.e)},{repr(self.f)},{repr(self.g)},{repr(self.h)}],'
                f'[{repr(self.i)},{repr(self.j)},{repr(self.k)},{repr(self.l)}],'
                f'[{repr(self.m)},{repr(self.n)},{repr(self.o)},{repr(self.p)}]]}}')

    """
    Custom methods
    """

    def transpose(self):
        return Matrix(self.a, self.e, self.i, self.m,
                      self.b, self.f, self.j, self.n,
                      self.c, self.g, self.k, self.o,
                      self.d, self.h, self.l, self.p)

    def aslist(self):
        return [self.a, self.b, self.c, self.d,
                self.e, self.f, self.g, self.h,
                self.i, self.j, self.k, self.l,
                self.m, self.n, self.o, self.p]

    def astuple(self):
        return (self.a, self.b, self.c, self.d,
                self.e, self.f, self.g, self.h,
                self.i, self.j, self.k, self.l,
                self.m, self.n, self.o, self.p)

    @staticmethod
    def zeros():
        return Matrix(0., 0., 0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 0.,
                      0., 0., 0., 0.)

    @staticmethod
    def ones():
        return Matrix(1., 1., 1., 1.,
                      1., 1., 1., 1.,
                      1., 1., 1., 1.,
                      1., 1., 1., 1.)

    @staticmethod
    def identity():
        return Matrix(1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.)

    # noinspection SpellCheckingInspection
    @staticmethod
    def fromrows(a: Vector, b: Vector, c: Vector, d: Vector):
        return Matrix(a.x, a.y, a.z, a.w,
                      b.x, b.y, b.z, b.w,
                      c.x, c.y, c.z, c.w,
                      d.x, d.y, d.z, d.w)

import math


class Vector:

    """
    A 4-dimensional vector, where the 4th element w is always 1
    """
    def __init__(self, x: (int, float, complex) = None, y: (int, float, complex) = None,
                 z: (int, float, complex) = None):
        self.x = x if x is not None else 0.
        self.y = y if y is not None else 0.
        self.z = z if z is not None else 0.
        self.w = 1.

    """
    Unary operators
    """
    def __pos__(self):
        """
        + operator
        """
        return Vector(self.x, self.y, self.z)

    def __neg__(self):
        """
        - operator
        """
        return Vector(-self.x, -self.y, -self.z)

    def __invert__(self):
        """
        ~ operator
        """
        return NotImplemented

    def __abs__(self):
        """
        abs()
        """
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def __floor__(self):
        """
        math.floor()
        """
        return Vector(math.floor(self.x), math.floor(self.y), math.floor(self.z))

    def __ceil__(self):
        """
        math.ceil()
        """
        return Vector(math.ceil(self.x), math.ceil(self.y), math.ceil(self.z))

    def __round__(self, n=None):
        """
        round()
        """
        return Vector(round(self.x, n), round(self.y, n), round(self.z, n))

    def __trunc__(self):
        """
        math.trunc()
        """
        return Vector(math.trunc(self.x), math.trunc(self.y), math.trunc(self.z))

    """
    Binary arithmetical operators
    """

    def __add__(self, other):
        """
        + operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return NotImplemented

    def __radd__(self, other):
        """
        + operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(other + self.x, other + self.y, other + self.z)
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        - operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """
        - operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(other - self.x, other - self.y, other - self.z)
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        * operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        * operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(other * self.x, other * self.y, other * self.z)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """
        @ operator
        """
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """
        @ operator
        """
        return NotImplemented

    def __pow__(self, power, modulo=None):
        """
        ** operator
        """
        if isinstance(power, int) or isinstance(power, float) or isinstance(power, complex):
            return Vector(self.x ** power, self.y ** power, self.z ** power)
        elif isinstance(power, Vector):
            return Vector(self.x ** power.x, self.y ** power.y, self.z ** power.z)
        else:
            return NotImplemented

    def __rpow__(self, other):
        """
        ** operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(other ** self.x, other ** self.y, other ** self.z)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        / operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        / operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Vector(other / self.x, other / self.y, other / self.z)
        else:
            return NotImplemented

    def __floordiv__(self, other):
        """
        // operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.x // other, self.y // other, self.z // other)
        elif isinstance(other, Vector):
            return Vector(self.x // other.x, self.y // other.y, self.z // other.z)
        else:
            return NotImplemented

    def __rfloordiv__(self, other):
        """
        // operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(other // self.x, other // self.y, other // self.z)
        else:
            return NotImplemented

    def __mod__(self, other):
        """
        % operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.x % other, self.y % other, self.z % other)
        elif isinstance(other, Vector):
            return Vector(self.x % other.x, self.y % other.y, self.z % other.z)
        else:
            return NotImplemented

    def __rmod__(self, other):
        """
        % operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(other % self.x, other % self.y, other % self.z)
        else:
            return NotImplemented

    def __divmod__(self, other):
        """
        divmod() function
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(divmod(self.x, other), divmod(self.y, other), divmod(self.z, other))
        elif isinstance(other, Vector):
            return Vector(divmod(self.x, other.x), divmod(self.y, other.y), divmod(self.z, other.z))
        else:
            return NotImplemented

    def __rdivmod__(self, other):
        """
        divmod() function
        """
        if isinstance(other, int) or isinstance(other, float):
            return Vector(divmod(other, self.x), divmod(other, self.y), divmod(other, self.z))
        else:
            return NotImplemented

    """
    Binary logical operators
    """
    def __eq__(self, other):
        """
        == operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self.x == other and self.y == other and self.z == other and self.w == other
        elif isinstance(other, Vector):
            return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        != operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self.x != other or self.y != other or self.z != other or self.w != other
        elif isinstance(other, Vector):
            return self.x != other.x or self.y != other.y or self.z != other.z or self.w != other.w
        else:
            return NotImplemented

    def __gt__(self, other):
        """
        > operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.x > other and self.y > other and self.z > other and self.w > other
        elif isinstance(other, Vector):
            return self.x > other.x and self.y > other.y and self.z > other.z and self.w > other.w
        else:
            return NotImplemented

    def __ge__(self, other):
        """
        >= operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.x >= other and self.y >= other and self.z >= other and self.w >= other
        elif isinstance(other, Vector):
            return self.x >= other.x and self.y >= other.y and self.z >= other.z and self.w >= other.w
        else:
            return NotImplemented

    def __lt__(self, other):
        """
        < operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.x < other and self.y < other and self.z < other and self.w < other
        elif isinstance(other, Vector):
            return self.x < other.x and self.y < other.y and self.z < other.z and self.w < other.w
        else:
            return NotImplemented

    def __le__(self, other):
        """
        <= operator
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.x <= other and self.y <= other and self.z <= other and self.w <= other
        elif isinstance(other, Vector):
            return self.x <= other.x and self.y <= other.y and self.z <= other.z and self.w <= other.w
        else:
            return NotImplemented

    """
    Inplace operators
    """
    def __iadd__(self, other):
        """
        += operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            self.x += other
            self.y += other
            self.z += other
            self.w += other
        elif isinstance(other, Vector):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            self.w += other.w
        else:
            return NotImplemented
        return self

    def __isub__(self, other):
        """
        -= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            self.x -= other
            self.y -= other
            self.z -= other
            self.w -= other
        elif isinstance(other, Vector):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
            self.w -= other.w
        else:
            return NotImplemented
        return self

    def __imul__(self, other):
        """
        *= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            self.x *= other
            self.y *= other
            self.z *= other
            self.w *= other
        elif isinstance(other, Vector):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
            self.w *= other.w
        else:
            return NotImplemented
        return self

    def __imatmul__(self, other):
        """
        @= operator
        """
        return NotImplemented

    def __ipow__(self, other):
        """
        **= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            self.x **= other
            self.y **= other
            self.z **= other
            self.w **= other
        elif isinstance(other, Vector):
            self.x **= other.x
            self.y **= other.y
            self.z **= other.z
            self.w **= other.w
        else:
            return NotImplemented
        return self

    def __itruediv__(self, other):
        """
        /= operator
        """
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            self.x /= other
            self.y /= other
            self.z /= other
            self.w /= other
        elif isinstance(other, Vector):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
            self.w /= other.w
        else:
            return NotImplemented
        return self

    def __ifloordiv__(self, other):
        """
        //= operator
        """
        if isinstance(other, int) or isinstance(other, float):
            self.x //= other
            self.y //= other
            self.z //= other
            self.w //= other
        elif isinstance(other, Vector):
            self.x //= other.x
            self.y //= other.y
            self.z //= other.z
            self.w //= other.w
        else:
            return NotImplemented
        return self

    def __imod__(self, other):
        """
        %= operator
        """
        if isinstance(other, int) or isinstance(other, float):
            self.x %= other
            self.y %= other
            self.z %= other
            self.w %= other
        elif isinstance(other, Vector):
            self.x %= other.x
            self.y %= other.y
            self.z %= other.z
            self.w %= other.w
        else:
            return NotImplemented
        return self

    """
    Other overloads
    """

    # noinspection PyRedundantParentheses
    def __str__(self):
        return (f'Vector{{[{str(self.x)},{str(self.y)},{str(self.z)},{str(self.w)}]}}')

    # noinspection PyRedundantParentheses
    def __repr__(self):
        return (f'Vector{{[{repr(self.x)},{repr(self.y)},{repr(self.z)},{repr(self.w)}]}}')

    """
    Custom methods
    """
    def norm(self):
        return math.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    # noinspection SpellCheckingInspection
    def aslist(self):
        return [self.x, self.y, self.z, self.w]

    # noinspection SpellCheckingInspection
    def astuple(self):
        return self.x, self.y, self.z, self.w

    @staticmethod
    def zeros():
        return Vector(0., 0., 0.)

    @staticmethod
    def ones():
        return Vector(1., 1., 1.)

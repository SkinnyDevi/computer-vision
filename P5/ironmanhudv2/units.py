from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point":
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Point":
        return Point(self.x / scalar, self.y / scalar)
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)
    
    def to_int_tuple(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))
    
    def scale(self, factor: float) -> "Point":
        return Point(self.x * factor, self.y * factor)
    
@dataclass
class Size:
    width: float
    height: float
    
    def __add__(self, other: "Size") -> "Size":
        return Size(self.width + other.width, self.height + other.height)
    
    def __sub__(self, other: "Size") -> "Size":
        return Size(self.width - other.width, self.height - other.height)
    
    def __mul__(self, scalar: float) -> "Size":
        return Size(self.width * scalar, self.height * scalar)
    
    def __truediv__(self, scalar: float) -> "Size":
        return Size(self.width / scalar, self.height / scalar)

    def to_tuple(self) -> tuple[float, float]:
        return (self.width, self.height)
    
    def to_int_tuple(self) -> tuple[int, int]:
        return (int(self.width), int(self.height))
    
    def scale(self, factor: float) -> "Size":
        return Size(self.width * factor, self.height * factor)
    
def point_to_size(point: Point) -> Size:
    return Size(point.x, point.y)

def size_to_point(size: Size) -> Point:
    return Point(size.width, size.height)
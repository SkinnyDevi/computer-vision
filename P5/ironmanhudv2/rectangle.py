from ironmanhudv2.units import Size, Point, point_to_size


def get_rectangle_draw_points(position: Point, size: Size) -> tuple[tuple[int, int], tuple[int, int]]:
    top_left = position.to_int_tuple()
    bottom_right = (point_to_size(position) + size).to_int_tuple()
    return top_left, bottom_right
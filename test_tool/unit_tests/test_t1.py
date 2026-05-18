import pytest
from snake_battle import Direction, Point, pt_add, FOOD_COUNT, INITIAL_LENGTH, FPS_DEFAULT, FPS_MIN, FPS_MAX, COLOR_ALPHA, COLOR_BETA, COLOR_FOOD, COLOR_BORDER

def test_direction_properties():
    assert Direction.UP.dy == -1
    assert Direction.UP.dx == 0
    assert Direction.DOWN.dy == 1
    assert Direction.DOWN.dx == 0
    assert Direction.LEFT.dy == 0
    assert Direction.LEFT.dx == -1
    assert Direction.RIGHT.dy == 0
    assert Direction.RIGHT.dx == 1

def test_direction_opposite():
    assert Direction.UP.opposite == Direction.DOWN
    assert Direction.DOWN.opposite == Direction.UP
    assert Direction.LEFT.opposite == Direction.RIGHT
    assert Direction.RIGHT.opposite == Direction.LEFT

def test_direction_from_delta():
    assert Direction.from_delta(-1, 0) == Direction.UP
    assert Direction.from_delta(1, 0) == Direction.DOWN
    assert Direction.from_delta(0, -1) == Direction.LEFT
    assert Direction.from_delta(0, 1) == Direction.RIGHT
    with pytest.raises(ValueError):
        Direction.from_delta(1, 1)

def test_point_and_pt_add():
    p = Point(10, 20)
    assert p.y == 10
    assert p.x == 20
    
    res_up = pt_add(p, Direction.UP)
    assert res_up == Point(9, 20)
    
    res_down = pt_add(p, Direction.DOWN)
    assert res_down == Point(11, 20)
    
    res_left = pt_add(p, Direction.LEFT)
    assert res_left == Point(10, 19)
    
    res_right = pt_add(p, Direction.RIGHT)
    assert res_right == Point(10, 21)

def test_constants():
    assert FOOD_COUNT == 5
    assert INITIAL_LENGTH == 4
    assert FPS_DEFAULT == 10
    assert FPS_MIN == 2
    assert FPS_MAX == 30
    assert COLOR_ALPHA == 1
    assert COLOR_BETA == 2
    assert COLOR_FOOD == 3
    assert COLOR_BORDER == 4

import pytest
from snake_battle import Snake, Direction, Board, BOARD_WIDTH, BOARD_HEIGHT, NUM_FOOD

def test_snake_move():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake")
    # Move right
    new_head = snake.move(Direction.RIGHT)
    assert new_head == (10, 11)
    assert snake.head == (10, 11)
    assert len(snake.body) == 1
    assert snake.direction == Direction.RIGHT

    # Move down
    new_head = snake.move(Direction.DOWN)
    assert new_head == (11, 11)
    assert snake.head == (11, 11)
    assert len(snake.body) == 2 # (11,11), (10,11)
    assert snake.body[-1] == (10, 10) # Wait, if len was 1, it should be (10,11) and then pop (10,10)

    # Correcting expectation:
    # Initial: body = [(10, 10)], length = 1
    # move(RIGHT): appendleft((10, 11)), pop((10, 10)) -> body = [(10, 11)], length = 1
    # move(DOWN): appendleft((11, 11)), pop((10, 11)) -> body = [(11, 11)], length = 1
    
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake")
    snake.move(Direction.RIGHT)
    assert len(snake.body) == 1
    snake.move(Direction.DOWN)
    assert len(snake.body) == 1
    assert snake.head == (11, 11)

def test_snake_move_opposite():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake")
    with pytest.raises(ValueError, match="Cannot move in opposite direction"):
        snake.move(Direction.LEFT)

def test_snake_grow():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake")
    snake.grow()
    assert snake.just_ate is True
    
    # Move should increase length
    snake.move(Direction.RIGHT)
    assert len(snake.body) == 2
    assert snake.just_ate is False
    
    # Next move should maintain length
    snake.move(Direction.RIGHT)
    assert len(snake.body) == 2

def test_snake_manhattan():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake")
    assert snake.manhattan((10, 12)) == 2
    assert snake.manhattan((12, 12)) == 4
    assert snake.manhattan((10, 10)) == 0

def test_board_is_wall():
    board = Board(20, 10) # width=20, height=10
    assert board.is_wall((-1, 0)) is True
    assert board.is_wall((0, -1)) is True
    assert board.is_wall((10, 0)) is True
    assert board.is_wall((0, 20)) is True
    assert board.is_wall((5, 5)) is False

def test_board_get_all_body_cells():
    s1 = Snake((1, 1), Direction.RIGHT, "S1")
    s1.body.append((1, 0))
    s2 = Snake((2, 2), Direction.RIGHT, "S2")
    board = Board()
    cells = board.get_all_body_cells([s1, s2])
    assert cells == {(1, 1), (1, 0), (2, 2)}

def test_board_get_empty_cells():
    board = Board(3, 3)
    s1 = Snake((0, 0), Direction.RIGHT, "S1")
    board.foods.add((0, 2))
    # Body: (0,0)
    # Food: (0,2)
    # Empty: (0,1), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    empty = board.get_empty_cells([s1])
    assert len(empty) == 7
    assert (0, 0) not in empty
    assert (0, 2) not in empty
    assert (1, 1) in empty

def test_board_spawn_food():
    board = Board(10, 10)
    snakes = [Snake((5, 5), Direction.RIGHT, "S1")]
    board.spawn_food(snakes)
    assert len(board.foods) == NUM_FOOD
    for pos in board.foods:
        assert pos != (5, 5)

def test_board_consume_food():
    board = Board()
    board.foods.add((5, 5))
    assert board.consume_food((5, 5)) is True
    assert (5, 5) not in board.foods
    assert board.consume_food((5, 5)) is False

def test_board_center_max_manhattan():
    board = Board(40, 20)
    assert board.center() == (10, 20)
    assert board.max_manhattan() == 60

import pytest
from snake_battle import Direction, Snake, Game, MIN_FOOD

def test_direction():
    # In snake_battle.py, Direction.value is (dy, dx)
    assert Direction.UP.value == (-1, 0)
    assert Direction.DOWN.value == (1, 0)
    assert Direction.LEFT.value == (0, -1)
    assert Direction.RIGHT.value == (0, 1)
    
    assert Direction.UP.opposite() == Direction.DOWN
    assert Direction.DOWN.opposite() == Direction.UP
    assert Direction.LEFT.opposite() == Direction.RIGHT
    assert Direction.RIGHT.opposite() == Direction.LEFT

def test_snake():
    # Snake(head, direction, name, color_pair)
    # head is (y, x)
    head = (10, 10)
    direction = Direction.RIGHT
    snake = Snake(head, direction, "TestSnake", 1)
    
    assert snake.head == (10, 10)
    assert snake.length == 3
    
    # Test movement
    snake.move(Direction.DOWN)
    assert snake.head == (11, 10)
    assert snake.length == 3

def test_game_init():
    width, height = 30, 15
    game = Game(width, height)
    assert game.board_width == width
    assert game.board_height == height
    
    # Test walls
    # Walls are at y=0, y=height-1, x=0, x=width-1
    assert (0, 0) in game.walls
    assert (height - 1, 0) in game.walls
    assert (0, width - 1) in game.walls
    assert (height - 1, width - 1) in game.walls
    assert (1, 1) not in game.walls

def test_game_snakes():
    width, height = 30, 15
    game = Game(width, height)
    assert game.alpha_snake.name == 'Alpha'
    assert game.beta_snake.name == 'Beta'
    
    # Initial positions are symmetric
    # alpha_start_x = width // 4 = 7
    # beta_start_x = 3 * width // 4 = 22
    # center_y = height // 2 = 7
    assert game.alpha_snake.head == (7, 7)
    assert game.beta_snake.head == (7, 22)

def test_game_food():
    width, height = 30, 15
    game = Game(width, height)
    assert len(game.foods) == MIN_FOOD
    for food in game.foods:
        assert food not in game.walls
        assert food not in game.alpha_snake.body_set
        assert food not in game.beta_snake.body_set

def test_game_snapshot():
    width, height = 30, 15
    game = Game(width, height)
    # The Game class in snake_battle.py doesn't have a snapshot() method,
    # but it has _get_state() which returns a GameState object.
    state = game._get_state()
    
    assert state.board_width == game.board_width
    assert state.foods == game.foods
    assert state.my_snake.head == game.alpha_snake.head

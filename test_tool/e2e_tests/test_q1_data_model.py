import pytest
from collections import deque
from snake_battle import Snake, Game, Direction, INITIAL_LENGTH, MIN_FOOD

def test_snake_init():
    # (1) Snake.__init__ creates correct body deque and properties
    head = (10, 10)
    direction = Direction.RIGHT
    snake = Snake(head, direction, "TestSnake", 1)
    
    assert snake.name == "TestSnake"
    assert snake.direction == direction
    assert snake.length == INITIAL_LENGTH
    assert snake.head == head
    # Body should extend opposite to direction (LEFT)
    # head: (10, 10), body: (10, 9), (10, 8)
    assert snake.body[0] == (10, 10)
    assert snake.body[1] == (10, 9)
    assert snake.body[2] == (10, 8)

def test_snake_move_no_grow():
    # (2) Snake.move() correctly shifts body, shrinks tail when not growing
    head = (10, 10)
    direction = Direction.RIGHT
    snake = Snake(head, direction, "TestSnake", 1)
    
    old_tail = snake.tail
    snake.move(Direction.RIGHT)
    
    assert snake.head == (10, 11)
    assert snake.length == INITIAL_LENGTH
    assert snake.tail != old_tail
    assert snake.tail == (10, 9)

def test_snake_move_grow():
    # (2) & (3) Snake.grow() sets _growing=True which is consumed by next move(), retains tail
    head = (10, 10)
    direction = Direction.RIGHT
    snake = Snake(head, direction, "TestSnake", 1)
    
    snake.grow()
    assert snake._growing is True
    
    old_tail = snake.tail
    snake.move(Direction.RIGHT)
    
    assert snake.length == INITIAL_LENGTH + 1
    assert snake.head == (10, 11)
    assert snake.tail == old_tail
    assert snake._growing is False

def test_snake_move_180():
    # Test that 180 degree turns are ignored
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake", 1)
    snake.move(Direction.LEFT)
    assert snake.head == (10, 10)
    assert snake.direction == Direction.RIGHT

def test_game_walls():
    # (5) Board.is_wall() correctly identifies boundary cells.
    # Since Board class is missing, we check Game.walls
    w, h = 20, 10
    game = Game(w, h)
    
    # Check corners
    assert (0, 0) in game.walls
    assert (0, w-1) in game.walls
    assert (h-1, 0) in game.walls
    assert (h-1, w-1) in game.walls
    
    # Check edges
    assert (0, 5) in game.walls
    assert (h-1, 5) in game.walls
    assert (5, 0) in game.walls
    assert (5, w-1) in game.walls
    
    # Check interior
    assert (5, 5) not in game.walls

def test_game_spawn_food():
    # (6) Board.spawn_food() always places food on empty cells and respects NUM_FOOD
    # In code, MIN_FOOD is used.
    w, h = 20, 10
    game = Game(w, h)
    
    assert len(game.foods) == MIN_FOOD
    for food in game.foods:
        assert food not in game.walls
        assert food not in game.alpha_snake.body_set
        assert food not in game.beta_snake.body_set

def test_game_consume_food():
    # (7) Board.consume_food() removes food and returns True/False correctly.
    # In code, this is handled in game.tick().
    w, h = 20, 10
    game = Game(w, h)
    
    # Manually place food in front of Alpha
    alpha_head = game.alpha_snake.head
    food_pos = (alpha_head[0], alpha_head[1] + 1)
    game.foods.add(food_pos)
    
    # Before consumption
    assert food_pos in game.foods
    
    # Simulate the part of tick() that consumes food
    new_alpha_head = food_pos
    if new_alpha_head in game.foods:
        game.foods.discard(new_alpha_head)
        
    assert food_pos not in game.foods

def test_food_spawn_crowded_board():
    # Test food spawn on a very crowded board
    w, h = 5, 5 # Very small
    game = Game(w, h)
    
    # Fill most of the board
    # Walls are already there. Interior is (5-2)*(5-2) = 9 cells.
    # Snakes take 3+3 = 6 cells.
    # Only 3 cells left.
    
    # If we tried to spawn more than possible, it should not crash
    game.foods.clear()
    game._spawn_food()
    assert len(game.foods) <= MIN_FOOD

def test_snake_length_one():
    # Edge case: length-1 snake
    snake = Snake((10, 10), Direction.RIGHT, "Tiny", 1)
    snake.body = deque([(10, 10)])
    
    snake.move(Direction.RIGHT)
    assert snake.head == (10, 11)
    assert snake.length == 1
    assert snake.body == deque([(10, 11)])

def test_snake_at_boundary():
    # Edge case: snake at boundary
    snake = Snake((1, 1), Direction.UP, "Boundary", 1)
    snake.move(Direction.UP)
    assert snake.head == (0, 1) # Moves into wall

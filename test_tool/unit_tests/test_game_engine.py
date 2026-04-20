import pytest
from snake_battle import Game, Direction, Snake, MIN_FOOD

def test_game_init():
    width, height = 30, 15
    game = Game(width, height)
    assert game.board_width == width
    assert game.board_height == height
    assert len(game.foods) == MIN_FOOD

def test_movement_180_turn():
    width, height = 30, 15
    game = Game(width, height)
    snake = game.alpha_snake
    initial_head = snake.head
    
    # Alpha starts facing RIGHT
    # Try to turn LEFT (opposite)
    snake.move(Direction.LEFT)
    
    # Should be ignored, snake remains facing RIGHT
    assert snake.direction == Direction.RIGHT
    # But wait, Snake.move() updates direction and then moves.
    # If it's a 180 turn, it returns immediately.
    # So head should not have changed.
    assert snake.head == initial_head

def test_movement_and_growth():
    width, height = 30, 15
    game = Game(width, height)
    snake = game.alpha_snake
    initial_len = snake.length
    
    # Force food to be in front of snake
    # head is (7, 7), direction is RIGHT, next head is (7, 8)
    food_pos = (7, 8)
    game.foods.clear()
    game.foods.add(food_pos)
    
    # Manually set growing to True to simulate eating
    # In game.tick(), if new_head in foods, snake._growing is set to True.
    # Then snake.move is called.
    # Wait, in snake_battle.py, the order in tick() is:
    # 1. New heads calculated
    # 2. move() called
    # 3. if grows, set _growing = True
    # 4. if _growing, body.append(tail)
    
    # To test Snake.move and growth:
    snake.grow()
    snake.move(Direction.RIGHT)
    
    assert snake.length == initial_len + 1

def test_wall_collision():
    width, height = 30, 15
    game = Game(width, height)
    # Move Alpha snake to wall
    # head is (7, 7), direction is RIGHT. 
    # Set head to (7, 1) and direction to LEFT
    game.alpha_snake.body.clear()
    game.alpha_snake.body.append((7, 1))
def test_wall_collision():
    width, height = 30, 15
    game = Game(width, height)
    # Mock AI to force a specific move
    game.alpha_ai.decide = lambda state: Direction.LEFT
    game.beta_ai.decide = lambda state: Direction.RIGHT
    
    # Move Alpha snake to wall
    game.alpha_snake.body.clear()
    game.alpha_snake.body.append((7, 1))
    game.alpha_snake.direction = Direction.LEFT
    
    # Now call tick.
    # AI will decide Direction.LEFT, new_alpha_head = (7, 1) + (0, -1) = (7, 0) which is a wall.
    game.tick()
    
    assert game.alpha_snake.alive is False
    assert game.game_over is True
    assert game.winner == "Beta"
    assert game.game_over is True

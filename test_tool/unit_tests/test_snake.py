import pytest
from snake_battle import Game, Snake, Direction, GameState, AIAlpha, AIBeta

def test_snake_initialization():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake", 1)
    assert snake.head == (10, 10)
    assert snake.length == 3
    assert snake.body[0] == (10, 10)
    assert snake.body[1] == (10, 9)
    assert snake.body[2] == (10, 8)

def test_snake_move():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake", 1)
    snake.move(Direction.RIGHT)
    assert snake.head == (10, 11)
    assert snake.length == 3
    assert snake.body[-1] == (10, 9)

def test_snake_move_invalid():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake", 1)
    # Try to move opposite
    snake.move(Direction.LEFT)
    assert snake.head == (10, 10)
    assert snake.direction == Direction.RIGHT

def test_snake_grow():
    snake = Snake((10, 10), Direction.RIGHT, "TestSnake", 1)
    snake.grow()
    snake.move(Direction.RIGHT)
    assert snake.length == 4
    assert snake.head == (10, 11)

def test_game_initialization():
    game = Game(30, 15)
    assert game.board_width == 30
    assert game.board_height == 15
    assert len(game.foods) == 5
    assert game.alpha_snake.head == (15 // 2, 30 // 4)
    assert game.beta_snake.head == (15 // 2, 3 * 30 // 4)

def test_game_tick_collision_wall():
    game = Game(30, 15, seed=42)
    # Force alpha to move into wall
    game.alpha_snake.direction = Direction.UP
    game.alpha_snake.body = [ (1, 1), (2, 1), (3, 1) ] # head at (1,1)
    # Force beta to move away
    game.beta_snake.body = [ (7, 7), (8, 7), (9, 7) ]
    
    # Alpha is at (1,1), moving UP goes to (0,1) which is a wall
    # We need to override AI decision to force the move
    # But Game.tick calls AIAIs' decide()
    # Let's mock the AI.
    
    class MockAI:
        def __init__(self, direction):
            self.direction = direction
        def decide(self, state):
            return self.direction
            
    game.alpha_ai = MockAI(Direction.UP)
    game.beta_ai = MockAI(Direction.RIGHT)
    
    game.tick()
    assert not game.alpha_snake.alive
    assert game.winner == "Beta"

def test_game_tick_collision_head_to_head():
    game = Game(30, 15, seed=42)
    # Place snakes head-to-head
    # Alpha: (7, 10) moving RIGHT -> (7, 11)
    # Beta: (7, 12) moving LEFT -> (7, 11)
    game.alpha_snake.body = [ (7, 10), (7, 9), (7, 8) ]
    game.alpha_snake.direction = Direction.RIGHT
    game.beta_snake.body = [ (7, 12), (7, 13), (7, 14) ]
    game.beta_snake.direction = Direction.LEFT
    
    class MockAI:
        def __init__(self, direction):
            self.direction = direction
        def decide(self, state):
            return self.direction
            
    game.alpha_ai = MockAI(Direction.RIGHT)
    game.beta_ai = MockAI(Direction.LEFT)
    
    game.tick()
    assert not game.alpha_snake.alive
    assert not game.beta_snake.alive
    assert game.winner == "Draw"

def test_game_tick_eating_food():
    game = Game(30, 15, seed=42)
    # Place snake and food
    game.alpha_snake.body = [ (7, 10), (7, 9), (7, 8) ]
    game.alpha_snake.direction = Direction.RIGHT
    game.foods = { (7, 11) }
    
    class MockAI:
        def __init__(self, direction):
            self.direction = direction
        def decide(self, state):
            return self.direction
            
    game.alpha_ai = MockAI(Direction.RIGHT)
    game.beta_ai = MockAI(Direction.RIGHT) # Beta moves away
    game.beta_snake.body = [ (7, 20), (7, 19), (7, 18) ]
    
    # Alpha moves from (7,10) to (7,11) and eats food
    game.tick()
    assert game.alpha_snake.length == 4
    assert (7, 11) not in game.foods

def test_game_tick_shrinking():
    game = Game(30, 15, seed=42)
    game.frame = 500 # SURVIVAL_GATE
    game.alpha_snake.body = [ (7, 10), (7, 11), (7, 12), (7, 13) ] # length 4
    game.beta_snake.body = [ (7, 20), (7, 21), (7, 22), (7, 23) ] # length 4
    
    # Mock AIs to move in directions that don't collide
    class MockAI:
        def __init__(self, direction):
            self.direction = direction
        def decide(self, state):
            return self.direction
            
    game.alpha_ai = MockAI(Direction.UP)
    game.beta_ai = MockAI(Direction.DOWN)
    
    # Frame 501: tick will set shrinking = True (if frame >= 500)
    # Game.tick() increments frame to 501.
    # Frame 501 % 30 == 501 % 30 = 21. No shrinking yet.
    # Shrinking happens when frame % 30 == 0.
    # Let's set frame to 539 so that 540 % 30 == 0.
    game.frame = 539
    game.tick() # Frame becomes 540. Shrinking happens.
    
    assert game.shrinking == True
    assert game.alpha_snake.length == 3
    assert game.beta_snake.length == 3

def test_ai_alpha_decision():
    game = Game(30, 15, seed=42)
    state = game._get_state()
    ai = AIAlpha(game.alpha_snake)
    
    # Check that it returns a valid Direction
    decision = ai.decide(state)
    assert isinstance(decision, Direction)

def test_ai_beta_decision():
    game = Game(30, 15, seed=42)
    state = game._get_state()
    ai = AIBeta(game.beta_snake)
    
    # Check that it returns a valid Direction
    decision = ai.decide(state)
    assert isinstance(decision, Direction)

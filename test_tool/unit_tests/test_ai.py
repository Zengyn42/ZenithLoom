import pytest
from snake_battle import Game, Snake, Direction, Pt, AIAlpha, AIBeta, GameState

def test_ai_alpha_decide():
    game = Game(30, 15)
    state = game._get_state()
    ai = AIAlpha(game.alpha_snake)
    move = ai.decide(state)
    assert isinstance(move, Direction)

def test_ai_beta_decide():
    game = Game(30, 15)
    state = game._get_state()
    ai = AIBeta(game.beta_snake)
    move = ai.decide(state)
    assert isinstance(move, Direction)

def test_snake_collision():
    # Create a game
    game = Game(30, 15)
    
    # Setup snakes head to head
    # Alpha head at (7, 10), Beta head at (7, 11)
    # Alpha moves RIGHT, Beta moves LEFT
    game.alpha_snake.body.clear()
    game.alpha_snake.body.append((7, 10))
    game.alpha_snake.body.append((7, 9))
    game.alpha_snake.body.append((7, 8))
    game.alpha_snake.direction = Direction.RIGHT
    
    game.beta_snake.body.clear()
    game.beta_snake.body.append((7, 11))
    game.beta_snake.body.append((7, 12))
    game.beta_snake.body.append((7, 13))
    game.beta_snake.direction = Direction.LEFT
    
    # Mock the AIs to move towards each other
    class MockAI:
        def __init__(self, dir):
            self.dir = dir
        def decide(self, state):
            return self.dir
            
    game.alpha_ai = MockAI(Direction.RIGHT)
    game.beta_ai = MockAI(Direction.LEFT)
    
    # One tick: they move to (7, 11) and (7, 10)
    # But wait, if they both move to the same spot?
    # Let's test head-to-head collision: both move to (7, 10.5) -> actually (7, 11) and (7, 10)
    
    # In snake_battle.py:
    # new_alpha_head = (7, 10) + (0, 1) = (7, 11)
    # new_beta_head = (7, 11) + (0, -1) = (7, 10)
    # This is a "crossover collision"
    
    game.tick()
    
    assert not game.alpha_snake.alive
    assert not game.beta_snake.alive
    assert game.winner == "Draw"

def test_head_on_collision():
    game = Game(30, 15)
    
    # Alpha at (7, 10), Beta at (7, 12)
    # Both move to (7, 11)
    game.alpha_snake.body.clear()
    game.alpha_snake.body.append((7, 10))
    game.alpha_snake.body.append((7, 9))
    game.alpha_snake.body.append((7, 8))
    
    game.beta_snake.body.clear()
    game.beta_snake.body.append((7, 12))
    game.beta_snake.body.append((7, 13))
    game.beta_snake.body.append((7, 14))
    
    class MockAI:
        def __init__(self, dir):
            self.dir = dir
        def decide(self, state):
            return self.dir
            
    game.alpha_ai = MockAI(Direction.RIGHT)
    game.beta_ai = MockAI(Direction.LEFT)
    
    game.tick()
    
    assert not game.alpha_snake.alive
    assert not game.beta_snake.alive
    assert game.winner == "Draw"

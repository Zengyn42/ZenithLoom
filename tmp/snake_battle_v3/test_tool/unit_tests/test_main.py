#!/usr/bin/env python3
"""Unit tests for Snake Battle v3."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snake_battle import (
    Direction, UP, DOWN, LEFT, RIGHT,
    Snake, GameState, AIAlpha, AIBeta, Game
)


class TestDirection:
    """Tests for Direction enum."""
    
    def test_directions_exist(self):
        """All cardinal directions exist."""
        assert Direction.UP is not None
        assert Direction.DOWN is not None
        assert Direction.LEFT is not None
        assert Direction.RIGHT is not None
    
    def test_direction_values(self):
        """Directions have correct (dy, dx) values."""
        assert Direction.UP.value == (-1, 0)
        assert Direction.DOWN.value == (1, 0)
        assert Direction.LEFT.value == (0, -1)
        assert Direction.RIGHT.value == (0, 1)
    
    def test_opposite(self):
        """Opposite direction is correct."""
        assert Direction.UP.opposite() == Direction.DOWN
        assert Direction.DOWN.opposite() == Direction.UP
        assert Direction.LEFT.opposite() == Direction.RIGHT
        assert Direction.RIGHT.opposite() == Direction.LEFT
    
    def test_dy_dx_properties(self):
        """Direction dy and dx properties are correct."""
        assert Direction.UP.dy == -1 and Direction.UP.dx == 0
        assert Direction.DOWN.dy == 1 and Direction.DOWN.dx == 0
        assert Direction.LEFT.dy == 0 and Direction.LEFT.dx == -1
        assert Direction.RIGHT.dy == 0 and Direction.RIGHT.dx == 1


class TestSnake:
    """Tests for Snake class."""
    
    def test_initialization(self):
        """Snake initializes with correct position and direction."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        assert snake.head == (10, 10)
        assert snake.direction == Direction.RIGHT
        assert snake.name == "Test"
        assert snake.alive is True
    
    def test_initial_body_length(self):
        """Snake starts with 3 body segments."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        assert snake.length == 3
    
    def test_body_segments_position(self):
        """Body segments are positioned correctly behind head."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        # Head at (10, 10), moving right, so body extends left
        assert (10, 10) in snake.body_set
        assert (10, 9) in snake.body_set
        assert (10, 8) in snake.body_set
    
    def test_move_in_direction(self):
        """Snake moves correctly in given direction."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        initial_head = snake.head
        
        snake.move(Direction.UP)
        assert snake.head == (9, 10)
        
        snake.move(Direction.RIGHT)
        assert snake.head == (9, 11)
    
    def test_ignores_opposite_direction(self):
        """Snake ignores 180-degree turns."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        initial_head = snake.head
        
        snake.move(Direction.LEFT)  # Should be ignored
        assert snake.head == initial_head
    
    def test_grow(self):
        """Snake grows when grow() is called and moved."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        initial_length = snake.length
        
        snake.grow()
        snake.move(Direction.UP)
        
        assert snake.length == initial_length + 1
    
    def test_body_set_property(self):
        """body_set property returns correct set."""
        snake = Snake((10, 10), Direction.RIGHT, "Test", 1)
        body_set = snake.body_set
        assert len(body_set) == 3
        assert (10, 10) in body_set


class TestGameState:
    """Tests for GameState class."""
    
    def test_initialization(self):
        """GameState initializes correctly."""
        my_snake = Snake((10, 10), Direction.RIGHT, "MySnake", 1)
        opponent = Snake((20, 20), Direction.LEFT, "Opponent", 2)
        foods = {(5, 5)}
        walls = set()
        
        state = GameState(
            my_snake=my_snake,
            opponent=opponent,
            foods=foods,
            walls=walls,
            board_width=20,
            board_height=20
        )
        
        assert state.my_snake is my_snake
        assert state.opponent is opponent
        assert state.foods == foods
        assert state.board_width == 20
        assert state.board_height == 20
    
    def test_interior_property(self):
        """interior property returns all non-wall positions."""
        my_snake = Snake((10, 10), Direction.RIGHT, "MySnake", 1)
        opponent = Snake((20, 20), Direction.LEFT, "Opponent", 2)
        foods = set()
        walls = set()
        
        state = GameState(
            my_snake=my_snake,
            opponent=opponent,
            foods=foods,
            walls=walls,
            board_width=10,
            board_height=10
        )
        
        interior = state.interior
        # Interior excludes walls (border), so it's (height-2) * (width-2)
        assert len(interior) == 8 * 8 == 64
        
        # Check that border is excluded
        assert (0, 0) not in interior
        assert (0, 5) not in interior
        assert (5, 0) not in interior
        assert (9, 5) not in interior
        assert (5, 9) not in interior
        
        # Check that interior includes inner cells
        assert (1, 1) in interior
        assert (5, 5) in interior
        assert (8, 8) in interior


class TestAIAlpha:
    """Tests for AIAlpha class."""
    
    def test_initialization(self):
        """AIAlpha initializes with snake."""
        snake = Snake((10, 10), Direction.RIGHT, "Alpha", 1)
        ai = AIAlpha(snake)
        assert ai.snake is snake
    
    def test_decide_returns_direction(self):
        """decide method returns a Direction."""
        snake = Snake((10, 10), Direction.RIGHT, "Alpha", 1)
        ai = AIAlpha(snake)
        
        state = GameState(
            my_snake=snake,
            opponent=Snake((20, 20), Direction.LEFT, "Beta", 2),
            foods={(5, 15)},
            walls=set(),
            board_width=30,
            board_height=30
        )
        
        result = ai.decide(state)
        assert isinstance(result, Direction)


class TestAIBeta:
    """Tests for AIBeta class."""
    
    def test_initialization(self):
        """AIBeta initializes with snake."""
        snake = Snake((10, 10), Direction.RIGHT, "Beta", 2)
        ai = AIBeta(snake)
        assert ai.snake is snake
    
    def test_decide_returns_direction(self):
        """decide method returns a Direction."""
        snake = Snake((10, 10), Direction.RIGHT, "Beta", 2)
        ai = AIBeta(snake)
        
        state = GameState(
            my_snake=snake,
            opponent=Snake((20, 20), Direction.LEFT, "Alpha", 1),
            foods={(5, 15)},
            walls=set(),
            board_width=30,
            board_height=30
        )
        
        result = ai.decide(state)
        assert isinstance(result, Direction)


class TestGame:
    """Tests for Game class."""
    
    def test_initialization(self):
        """Game initializes correctly."""
        game = Game(width=50, height=30, seed=42)
        
        assert game.board_width == 50
        assert game.board_height == 30
        assert game.frame == 0
        assert game.game_over is False
        assert game.winner is None
        assert game.alpha_snake.alive is True
        assert game.beta_snake.alive is True
    
    def test_snares_positioned_correctly(self):
        """Snakes are positioned on opposite sides."""
        game = Game(width=50, height=30, seed=42)
        
        mid_y = 30 // 2
        left_x = 50 // 4
        right_x = 3 * 50 // 4
        
        assert game.alpha_snake.head == (mid_y, left_x)
        assert game.beta_snake.head == (mid_y, right_x)
        assert game.alpha_snake.direction == Direction.RIGHT
        assert game.beta_snake.direction == Direction.LEFT
    
    def test_tick_advances_frame(self):
        """tick() method advances the frame counter."""
        game = Game(width=50, height=30, seed=42)
        
        game.tick()
        assert game.frame == 1
        
        game.tick()
        assert game.frame == 2
    
    def test_foods_exist(self):
        """Game has foods on initialization."""
        game = Game(width=50, height=30, seed=42)
        assert len(game.foods) > 0
    
    def test_walls_exist(self):
        """Game has walls on initialization."""
        game = Game(width=50, height=30, seed=42)
        assert len(game.walls) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

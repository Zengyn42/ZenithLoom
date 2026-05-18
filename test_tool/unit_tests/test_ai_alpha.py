import pytest
from snake_battle import AIAlpha, Snake, Board, Direction, get_obstacles, voronoi_space, bfs_distances, nearest_foods
import collections

def test_ai_alpha_greedy_food():
    board = Board(10, 10)
    board.foods = {(2, 2)}
    snake = Snake("Alpha", [(0, 0)], Direction.RIGHT, 1)
    enemy = Snake("Enemy", [(5, 5)], Direction.RIGHT, 2)
    ai = AIAlpha(snake)
    
    # Mock current direction to allow moving DOWN or RIGHT
    snake.direction = Direction.LEFT 
    # Candidates: UP, DOWN, LEFT (opposite), RIGHT. Wait, opposite of LEFT is RIGHT.
    # Candidates: UP, DOWN, LEFT.
    
    # Setup: head at (0,0), food at (2,2).
    # Candidates: RIGHT (opposite of LEFT), UP, DOWN, LEFT.
    # Safe dirs will be filtered.
    # Let's just set direction so RIGHT is a candidate.
    snake.direction = Direction.UP # Candidates: LEFT, RIGHT, DOWN
    
    # Food is at (2,2). Direction RIGHT (0,1) or DOWN (1,0) would both lead closer.
    # Let's check if it picks one.
    dir_chosen = ai.choose(board, enemy)
    assert dir_chosen in [Direction.RIGHT, Direction.DOWN]

def test_ai_alpha_safe_directions():
    board = Board(5, 5)
    snake = Snake("Alpha", [(0, 0)], Direction.RIGHT, 1)
    enemy = Snake("Enemy", [(0, 1)], Direction.RIGHT, 2) # Enemy is right next to head
    ai = AIAlpha(snake)
    
    candidates = [Direction.RIGHT, Direction.DOWN]
    # (0,1) is enemy head -> unsafe
    # (1,0) is empty -> safe (if voronoi space is okay)
    
    safe = ai._safe_directions(candidates, board, enemy)
    assert Direction.RIGHT not in safe
    assert Direction.DOWN in safe

def test_ai_alpha_intercept_mode():
    board = Board(10, 10)
    board.foods = {(5, 5)}
    # Alpha is longer
    snake = Snake("Alpha", [(0, 0)] * 10, Direction.RIGHT, 1)
    # Enemy is shorter
    enemy = Snake("Enemy", [(6, 5)] * 2, Direction.RIGHT, 2)
    ai = AIAlpha(snake)
    
    # Alpha food distance: (0,0) to (5,5) = 10
    # Enemy food distance: (6,5) to (5,5) = 1
    # Alpha is not closer to the food, so intercept should not trigger for this food.
    # Let's move enemy further.
    enemy.body = collections.deque([(9, 9)] * 2)
    
    # Alpha head (0,0), food (5,5), enemy (9,9)
    # Alpha is closer to food (5,5) than enemy (9,9).
    # But intercept checks if alpha_dist <= enemy_dist.
    
    safe_dirs = [Direction.RIGHT, Direction.DOWN]
    res = ai._intercept_mode(safe_dirs, board, enemy)
    assert res is not None

def test_ai_alpha_tail_chase():
    board = Board(10, 10)
    # Snake body: head at (0,0), tail at (0,2)
    snake = Snake("Alpha", [(0, 0), (0, 1), (0, 2)], Direction.RIGHT, 1)
    enemy = Snake("Enemy", [(5, 5)], Direction.RIGHT, 2)
    ai = AIAlpha(snake)
    
    # Tail is at (0,2). Head is at (0,0).
    # Direction to (0,2) from (0,0) is RIGHT.
    res = ai._tail_chase(board, enemy)
    assert res == Direction.RIGHT

def test_ai_alpha_fallback():
    board = Board(3, 3)
    # Trap Alpha in a corner
    snake = Snake("Alpha", [(0, 0)], Direction.RIGHT, 1)
    # Enemy occupies (0,1) and (1,0)
    enemy = Snake("Enemy", [(0, 1), (1, 0)], Direction.RIGHT, 2)
    ai = AIAlpha(snake)
    
    # No safe moves
    # candidates: UP, DOWN, LEFT (if dir was RIGHT), but opposite of RIGHT is LEFT.
    # So candidates: UP, DOWN, RIGHT.
    # But (0,1) is enemy and (1,0) is enemy and (-1,0) is out of bounds.
    # Should fallback to first candidate or current direction.
    
    res = ai.choose(board, enemy)
    assert isinstance(res, Direction)

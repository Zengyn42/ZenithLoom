
import pytest
from snake_battle import Direction, Point, pt_add, AIBeta

class MockSnake:
    def __init__(self, body):
        self.body = body
    
    @property
    def head(self):
        return self.body[0]
    
    def body_set(self):
        return set(self.body)

class MockGame:
    def __init__(self, alpha, beta, foods, wall_cells, H, W):
        self.alpha = alpha
        self.beta = beta
        self.foods = foods
        self.wall_cells = wall_cells
        self.H = H
        self.W = W

def test_ai_beta_food_selection():
    # Setup a scenario where beta is closer to one food than alpha,
    # and another food is closer to alpha.
    # Beta head at (2, 2), Alpha head at (2, 4)
    # Food 1 at (2, 1) - Beta is closer
    # Food 2 at (2, 5) - Alpha is closer
    beta_snake = MockSnake([Point(2, 2), Point(2, 3)])
    alpha_snake = MockSnake([Point(2, 4), Point(2, 5)])
    foods = [Point(2, 1), Point(2, 5)] # Note: alpha is on top of Food 2, but it's still a food
    wall_cells = set()
    game = MockGame(alpha_snake, beta_snake, foods, wall_cells, 10, 10)
    
    ai_beta = AIBeta(beta_snake)
    direction = ai_beta.choose(game)
    
    # Beta should go for Food 1 at (2, 1), which is LEFT from (2, 2)
    assert direction == Direction.LEFT

def test_ai_beta_contested_food():
    # Beta head at (2, 2), Alpha head at (2, 4)
    # Food 1 at (2, 3) - Beta: dist 1, Alpha: dist 1
    # Food 2 at (2, 1) - Beta: dist 1, Alpha: dist 3
    # Score for Food 1: 1 + 2 * max(0, 1-1) = 1
    # Score for Food 2: 1 + 2 * max(0, 1-3) = 1
    # Wait, the score formula is: score = my_dist + 2 * max(0, my_dist - opp_dist)
    # Food 1: my=1, opp=1 => 1 + 2*0 = 1
    # Food 2: my=1, opp=3 => 1 + 2*0 = 1
    # They have the same score. Tie-break is my_dist. Both are 1.
    # Let's make Food 2 slightly further.
    # Food 1: my=1, opp=1 => 1
    # Food 3: my=2, opp=10 => 2 + 2*0 = 2
    # Beta should pick Food 1.
    
    beta_snake = MockSnake([Point(2, 2)])
    alpha_snake = MockSnake([Point(2, 4)])
    foods = [Point(2, 3), Point(2, 0)] 
    # Food 1 (2, 3): beta dist 1, alpha dist 1. Score = 1 + 2*0 = 1
    # Food 2 (2, 0): beta dist 2, alpha dist 4. Score = 2 + 2*0 = 2
    
    game = MockGame(alpha_snake, beta_snake, foods, set(), 10, 10)
    ai_beta = AIBeta(beta_snake)
    direction = ai_beta.choose(game)
    
    # Should go for (2, 3) which is RIGHT
    assert direction == Direction.RIGHT

def test_ai_beta_penalize_contested():
    # Beta head at (2, 2), Alpha head at (2, 3)
    # Food 1: Beta dist 1, Alpha dist 0 (alpha is on it)
    # Food 2: Beta dist 2, Alpha dist 10
    # Food 1: score = 1 + 2 * max(0, 1-0) = 3
    # Food 2: score = 2 + 2 * max(0, 2-10) = 2
    # Beta should prefer Food 2 even if it's further, because Food 1 is contested/closer to alpha.
    
    beta_snake = MockSnake([Point(2, 2)])
    alpha_snake = MockSnake([Point(2, 3)])
    foods = [Point(2, 3), Point(2, 0)] # (2,0) is dist 2 from beta, dist 5 from alpha
    
    game = MockGame(alpha_snake, beta_snake, foods, set(), 10, 10)
    ai_beta = AIBeta(beta_snake)
    direction = ai_beta.choose(game)
    
    # Should go for (2, 0) which is LEFT
    assert direction == Direction.LEFT

def test_ai_beta_fallback_tail():
    # No reachable foods, should try to go to tail
    beta_snake = MockSnake([Point(1, 1), Point(1, 2), Point(0, 2)])
    alpha_snake = MockSnake([Point(5, 5)])
    foods = [] # No foods
    game = MockGame(alpha_snake, beta_snake, foods, set(), 10, 10)
    
    ai_beta = AIBeta(beta_snake)
    direction = ai_beta.choose(game)
    
    # Path from (1,1) to (0,2): (1,1) -> (0,1) -> (0,2) or (1,1) -> (1,2) -> (0,2)
    # Let's check what path_to_tail_direction returns.
    # It uses BFS. 
    # Tail is at (0, 2). Head is at (1, 1).
    # Possible steps from (1,1): UP (0,1), DOWN (2,1), LEFT (1,0), RIGHT (1,2)
    # (0,1) is dist 1 to (0,2). (1,2) is dist 1 to (0,2).
    # BFS will return the first one it finds.
    assert direction in [Direction.UP, Direction.RIGHT]

def test_ai_beta_fallback_flood():
    # No foods, no path to tail, should use flood fill
    # Create a box around beta head, but one direction is more open.
    # Head at (1,1). Walls at (0,1) and (2,1).
    # (1,0) is a dead end. (1,2) leads to a large open area.
    beta_snake = MockSnake([Point(1, 1)])
    alpha_snake = MockSnake([Point(5, 5)])
    foods = []
    wall_cells = {Point(0, 1), Point(2, 1), Point(1, 0)} # Right is open
    # Note: if we block (1,0), the only safe dir is RIGHT (1,2)
    # To actually test flood fill, let's have two options:
    # Option 1: Right -> (1,2) -> only 1 cell
    # Option 2: Down -> (2,1) -> many cells
    # Wait, I'll just make sure it picks the one with more space.
    
    # Let's refine:
    # Head (1,1). 
    # UP: (0,1) - wall
    # DOWN: (2,1) - open, leads to (2,2), (3,1), (3,2)
    # LEFT: (1,0) - open, dead end
    # RIGHT: (1,2) - wall
    
    # But our walls must include snake bodies.
    # To simplify, let's just see if it picks the best flood among safe dirs.
    
    beta_snake = MockSnake([Point(1, 1)])
    alpha_snake = MockSnake([Point(5, 5)])
    foods = []
    # Force a situation where tail path is not available (snake length 1)
    # and only certain directions are safe.
    wall_cells = {Point(0, 1), Point(1, 2)} # Only LEFT and DOWN are safe
    # LEFT (1,0) is a dead end: walls at (0,0), (2,0), (1,0) - no, (1,0) is the cell.
    # Let's put walls around (1,0) so its flood count is 1.
    # DOWN (2,1) is open.
    wall_cells.update({Point(0, 0), Point(2, 0), Point(1, -1)}) # Not possible with 0 <= x < W
    # Let's just use a small grid.
    # Head (1,1). H=3, W=3.
    # UP (0,1): Wall
    # RIGHT (1,2): Wall
    # LEFT (1,0): Safe, but (0,0) and (2,0) are walls. Flood = 1
    # DOWN (2,1): Safe, (2,0) and (2,2) are open. Flood = 3
    
    wall_cells = {Point(0, 1), Point(1, 2), Point(0, 0), Point(2, 0)}
    # wait, (2,0) is shared.
    # Let's just use a simple wall set.
    wall_cells = {Point(0, 1), Point(1, 2), Point(0, 0)} 
    # (1,0) safe. Neighbors of (1,0): (0,0)-wall, (2,0)-open, (1,1)-visited.
    # (2,1) safe. Neighbors of (2,1): (2,0)-open, (2,2)-open, (1,1)-visited.
    
    game = MockGame(alpha_snake, beta_snake, foods, wall_cells, 3, 3)
    ai_beta = AIBeta(beta_snake)
    direction = ai_beta.choose(game)
    
    # (2,1) should have a larger flood count than (1,0)
    assert direction == Direction.DOWN

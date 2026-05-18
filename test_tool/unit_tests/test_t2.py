import pytest
from collections import deque
from snake_battle import Snake, get_obstacles, danger_zone, flood_count, UP, DOWN, LEFT, RIGHT

def test_get_obstacles():
    # Setup
    rows, cols = 5, 5
    # Snake A: head (2,2), body [(2,2), (2,1)], grow_pending=0
    snake_a = Snake((2,1), UP, 1)
    snake_a.body = deque([(2,1), (2,2)]) # Note: head is (2,1)
    snake_a.direction = UP
    snake_a.grow_pending = 0
    
    # Snake B: head (2,3), body [(2,3), (3,3)], grow_pending=1
    snake_b = Snake((3,3), UP, 2)
    snake_b.body = deque([(2,3), (3,3)]) # Note: head is (2,3)
    snake_b.direction = UP
    snake_b.grow_pending = 1
    
    # Walls: r=0, r=4, c=0, c=4
    # Snake A: grow_pending=0, so exclude tail (2,2). Only (2,1) is obstacle.
    # Snake B: grow_pending=1, include all: (2,3), (3,3).
    
    obstacles = get_obstacles(snake_a, snake_b, rows, cols)
    
    # Walls
    assert (0, 0) in obstacles
    assert (0, 2) in obstacles
    assert (4, 0) in obstacles
    assert (2, 0) in obstacles
    assert (2, 4) in obstacles
    
    # Snake A
    assert (2, 1) in obstacles
    assert (2, 2) not in obstacles
    
    # Snake B
    assert (2, 3) in obstacles
    assert (3, 3) in obstacles

def test_danger_zone():
    snake_b = Snake((2,2), UP, 2)
    snake_b.direction = UP
    # Opponent can go LEFT, RIGHT, or stay UP (not DOWN)
    # UP: (1,2), LEFT: (2,1), RIGHT: (2,3)
    obstacles = set()
    dz = danger_zone(snake_b, obstacles)
    assert (1, 2) in dz
    assert (2, 1) in dz
    assert (2, 3) in dz
    assert (3, 2) not in dz # opposite to UP
    
    # Test obstacles blocking danger zone
    obstacles = {(1, 2)}
    dz = danger_zone(snake_b, obstacles)
    assert (1, 2) not in dz
    assert (2, 1) in dz
    assert (2, 3) in dz

def test_flood_count():
    # Simple 3x3 open space
    obstacles = set()
    # Start at (1,1)
    # Reachable: (1,1), (1,0), (1,2), (0,1), (2,1) ...
    count = flood_count((1,1), obstacles, 100)
    # Since there are no obstacles, it should count all reachable? 
    # No, the board is infinite in this function.
    # But with limit 10, it should return 10.
    assert flood_count((1,1), obstacles, 10) == 10
    
    # With constraints
    # (1,1) is start. Obstacles at (1,0), (0,1)
    # Reachable from (1,1): (1,1), (1,2), (2,1)
    # Then from (1,2): (0,2), (2,2)
    # Then from (2,1): (2,0), (2,2)
    obstacles = {(1,0), (0,1)}
    # Starting at (1,1):
    # 1. (1,1)
    # 2. (1,2)
    # 3. (2,1)
    # 4. (0,2) [from (1,2)]
    # 5. (2,2) [from (1,2) or (2,1)]
    # 6. (2,0) [from (2,1)]
    # All others...
    
    # Let's use a more constrained example
    # 3x3 box, walls at boundary
    obstacles = set()
    for r in range(5):
        for c in range(5):
            if r == 0 or r == 4 or c == 0 or c == 4:
                obstacles.add((r,c))
    
    # Center is (2,2). 3x3 inner area = 9 cells.
    count = flood_count((2,2), obstacles, 100)
    assert count == 9
    
    # Limit test
    count = flood_count((2,2), obstacles, 5)
    assert count == 5

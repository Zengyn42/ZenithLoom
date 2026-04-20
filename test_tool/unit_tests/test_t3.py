import pytest
from snake_battle import Pt, Direction, Snake, GameBoard, manhattan, safe_directions, temporal_flood_fill, a_star, voronoi_partition

def test_manhattan():
    assert manhattan(Pt(0, 0), Pt(3, 4)) == 7
    assert manhattan(Pt(1, 1), Pt(1, 1)) == 0
    assert manhattan(Pt(10, 5), Pt(5, 10)) == 10

def test_safe_directions():
    board = GameBoard(10, 10)
    # Initial snake body: (2, 5), (1, 5), (0, 5) - Wait, walls are at edges.
    # Let's create a board with larger size to avoid walls.
    board = GameBoard(20, 20)
    snake = Snake([(5, 5), (4, 5), (3, 5)], Direction.RIGHT, "S1", 1)
    board.snakes = [snake]
    
    # Head at (5, 5), direction RIGHT.
    # Opposite is LEFT (4, 5) - should be blocked.
    # UP (5, 4), DOWN (5, 6), RIGHT (6, 5) should be safe.
    safe = safe_directions(snake, board)
    assert Direction.RIGHT in safe
    assert Direction.UP in safe
    assert Direction.DOWN in safe
    assert Direction.LEFT not in safe

    # Test wall restriction
    # Put snake head at (1, 1), direction RIGHT.
    # UP (1, 0) is a wall.
    snake.body = deque([Pt(1, 1), Pt(1, 2)]) 
    # Wait, Snake class uses deque, but I initialized with list. Let's redefine carefully.
    from collections import deque
    snake = Snake([(1, 1)], Direction.RIGHT, "S1", 1)
    board.snakes = [snake]
    # Pt(1, 0) is a wall.
    safe = safe_directions(snake, board)
    assert Direction.UP not in safe

    # Test snake body collision
    # Snake head at (5, 5), body at (6, 5)
    snake = Snake([(5, 5), (6, 5)], Direction.UP, "S1", 1)
    board.snakes = [snake]
    # RIGHT is (6, 5) - should be blocked.
    safe = safe_directions(snake, board)
    assert Direction.RIGHT not in safe

    # Test tail moving (grow_counter == 0)
    # Snake head (5, 5), body (5, 6), (5, 7). Tail is (5, 7).
    snake = Snake([(5, 5), (5, 6), (5, 7)], Direction.UP, "S1", 1)
    snake.grow_counter = 0
    board.snakes = [snake]
    # Move DOWN to (5, 6) - blocked.
    # Move to (5, 7)? No, wait.
    # Need the tail to be at the target.
    # Snake: (5, 5), (5, 6), (5, 7). Body[0]=5,5 Body[1]=5,6 Body[2]=5,7 (tail)
    # He is at (5, 5), moving DOWN to (5, 6) is blocked.
    # Let's put tail at (6, 5) and head at (5, 5).
    snake = Snake([(5, 5), (6, 5)], Direction.UP, "S1", 1)
    snake.grow_counter = 0
    board.snakes = [snake]
    # Moving RIGHT to (6, 5) should be safe because tail (6, 5) will move.
    safe = safe_directions(snake, board)
    assert Direction.RIGHT in safe

    # Test grow_counter > 0 (tail doesn't move)
    snake.grow_counter = 1
    safe = safe_directions(snake, board)
    assert Direction.RIGHT not in safe

    # Test forced death
    # Box the snake in.
    board.walls.add(Pt(6, 5))
    board.walls.add(Pt(5, 4))
    board.walls.add(Pt(5, 6))
    # Direction UP (5, 4) - wall
    # Direction DOWN (5, 6) - wall
    # Direction RIGHT (6, 5) - wall
    # Direction LEFT (4, 5) - opposite (if length > 1)
    snake = Snake([(5, 5), (4, 5)], Direction.RIGHT, "S1", 1)
    board.snakes = [snake]
    safe = safe_directions(snake, board)
    # Should return all except opposite
    assert Direction.UP in safe
    assert Direction.DOWN in safe
    assert Direction.RIGHT in safe
    assert Direction.LEFT not in safe

def test_temporal_flood_fill():
    board = GameBoard(10, 10) # walls at 0 and 9
    snake = Snake([(2, 2), (2, 3)], Direction.UP, "S1", 1)
    snake.grow_counter = 0
    snakes = [snake]
    foods = set()
    
    # Start BFS from (3, 2)
    res = temporal_flood_fill(board, Pt(3, 2), snakes, foods)
    
    assert Pt(3, 2) in res
    assert res[Pt(3, 2)] == 0
    assert Pt(3, 3) in res
    assert res[Pt(3, 3)] == 1
    
    # Check if snake body blocks.
    # Snake body: (2, 2) - ttl 1, (2, 3) - ttl 2.
    # Path to (2, 2) from (3, 2) is depth 1. 
    # Arrival (1) >= decay_map.get((2, 2), 0) (1).
    # This logic is a bit strange: "If arrival >= cell_ttl: add to visited".
    # Usually you want arrival < ttl to be safe.
    # "Check nh not in any snake's body (BUT allow if nh == tail...)" in safe_directions.
    # But in temporal_flood_fill: "If arrival >= cell_ttl: add to visited".
    # Let's re-read the prompt: "If arrival >= cell_ttl: add to visited and queue."
    # Wait, that means it ONLY visits it if the snake has already vacated it?
    # Yes, that's what that means.
    assert Pt(2, 2) in res
    # Arrival at (2, 2) is 1. ttl for (2, 2) is 1. 1 >= 1 is True.
    
def test_a_star():
    walls = {Pt(1, 0), Pt(1, 1)}
    obstacles = {Pt(2, 1)}
    start = Pt(0, 0)
    goal = Pt(2, 0)
    # Path: (0,0) -> (0,1) -> (1,1) (wall) NO
    # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) (obstacle) NO
    # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) is blocked.
    # Try: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) is blocked.
    # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) block.
    # Let's use a simpler one.
    
    # (0,0) to (2,0) with wall at (1,0)
    walls = {Pt(1, 0)}
    obstacles = set()
    path = a_star(Pt(0, 0), Pt(2, 0), obstacles, walls, 10, 10)
    assert path == [Pt(0, 0), Pt(0, 1), Pt(1, 1), Pt(2, 1), Pt(2, 0)]
    
    # No path
    walls = {Pt(1, 0), Pt(0, 1)}
    path = a_star(Pt(0, 0), Pt(2, 0), obstacles, walls, 10, 10)
    assert path == []

def test_voronoi_partition():
    board = GameBoard(10, 10)
    s1 = Snake([(2, 5)], Direction.RIGHT, "S1", 1)
    s2 = Snake([(7, 5)], Direction.LEFT, "S2", 1)
    board.snakes = [s1, s2]
    foods = set()
    
    c1, c2 = voronoi_partition(board, board.snakes, foods)
    assert c1 > 0
    assert c2 > 0
    assert c1 + c2 <= 8*8 # Internal area

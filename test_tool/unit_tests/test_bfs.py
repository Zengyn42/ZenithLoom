import collections
import pytest
from snake_battle import Board, bfs_distances, bfs_path_exists, voronoi_space, nearest_foods

def test_bfs_distances():
    board = Board(5, 5)
    start = (2, 2)
    occupied = {(2, 1), (2, 3), (1, 2), (3, 2)} # Wall around start
    # Should only see the start itself
    dist = bfs_distances(start, occupied, board)
    assert dist == {(2, 2): 0}

    occupied = set()
    dist = bfs_distances(start, occupied, board)
    assert dist[(0, 0)] == 4
    assert dist[(4, 4)] == 4
    assert len(dist) == 25

def test_bfs_path_exists():
    board = Board(5, 5)
    start = (0, 0)
    goal = (4, 4)
    occupied = set()
    assert bfs_path_exists(start, goal, occupied, board) is True
    
    # Create a wall separating start and goal
    # Wall at x=2
    occupied = {(y, 2) for y in range(5)}
    assert bfs_path_exists(start, goal, occupied, board) is False

def test_voronoi_space():
    board = Board(5, 5)
    head_a = (0, 0)
    head_b = (0, 4)
    occupied = set()
    # A is at (0,0), B is at (0,4)
    # Midpoint is (0,2). 
    # Distance from (0,0) to (0,0) is 0, to (0,1) is 1, to (0,2) is 2.
    # Distance from (0,4) to (0,4) is 0, to (0,3) is 1, to (0,2) is 2.
    # (0,2) should be a tie.
    
    # For a 5x5 board, A and B are at opposite corners of the top row.
    # X-coordinates: 0, 1, 2, 3, 4
    # A is at 0, B is at 4.
    # Col 0, 1 are closer to A. Col 3, 4 are closer to B.
    # Col 2 is equidistance from A and B?
    # (0,2) is 2 away from both.
    # (1,2) is sqrt(1^2 + 2^2) - no, Manhattan distance.
    # (1,2) is |1-0| + |2-0| = 3 from A, and |1-0| + |2-4| = 3 from B.
    # So all cells in Col 2 should be ties.
    
    count_a, count_b = voronoi_space(head_a, head_b, occupied, board)
    # Col 0: 5 cells, Col 1: 5 cells => 10 for A
    # Col 3: 5 cells, Col 4: 5 cells => 10 for B
    # Col 2: 5 cells => Tie
    assert count_a == 10
    assert count_b == 10

    # Test with obstacle
    occupied = {(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)} # Wall at x=1
    # Now A is trapped in Col 0.
    # B can reach Col 1-4.
    count_a, count_b = voronoi_space(head_a, head_b, occupied, board)
    assert count_a == 5
    assert count_b == 15 # Col 2, 3, 4 (15 cells) minus the wall? Wait.
    # Col 1 is occupied. So B takes Col 2, 3, 4.
    # B is at (0, 4). 
    # Distances to (y, 2): y + |2-4| = y + 2.
    # (0, 2) is 2. (1, 2) is 3. (2, 2) is 4. (3, 2) is 5. (4, 2) is 6.
    # A is at (0, 0). It can't pass the wall.
    # So B should get everything in Col 2, 3, 4.
    # Total cells = 25. Wall = 5. A = 5. B = 15. Correct.

def test_nearest_foods():
    board = Board(10, 10)
    head = (5, 5)
    foods = {(5, 6), (5, 4), (6, 5), (4, 5), (0, 0)}
    occupied = set()
    
    # 4 foods are at distance 1, 1 food is far.
    # Should return 3 of the nearest ones.
    nearest = nearest_foods(head, foods, occupied, board, n=3)
    assert len(nearest) == 3
    for food, dist in nearest:
        assert dist == 1

    # Test with obstacles
    occupied = {(5, 6), (4, 5)} # Block two of the distance-1 foods
    # Remaining foods: (5, 4) dist 1, (6, 5) dist 1, (0, 0) dist 10
    nearest = nearest_foods(head, foods, occupied, board, n=3)
    assert len(nearest) == 3
    # One is (5,4) dist 1, one is (6,5) dist 1, one is (0,0) dist 10
    dists = sorted([d for f, d in nearest])
    assert dists == [1, 1, 10]

import pytest
from collections import deque
from snake_battle import Point, Direction, bfs_dist, bfs_first_step, flood_count, path_to_tail_direction, Snake

def test_bfs_dist():
    obstacles = {Point(1, 1), Point(1, 2), Point(2, 1)}
    targets = {Point(2, 2)}
    start = Point(0, 0)
    # (0,0) -> (0,1) -> (0,2) -> (1,2) [obs]
    # (0,0) -> (1,0) -> (2,0) -> (2,1) [obs]
    # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) is blocked.
    # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) is blocked.
    # Let's use a simpler one.
    obstacles = {Point(0, 1), Point(1, 0)}
    targets = {Point(1, 1)}
    start = Point(0, 0)
    assert bfs_dist(start, targets, obstacles) == float('inf')
    
    obstacles = set()
    targets = {Point(1, 1)}
    start = Point(0, 0)
    assert bfs_dist(start, targets, obstacles) == 2

def test_bfs_first_step():
    obstacles = set()
    start = Point(0, 0)
    target = Point(0, 2)
    # Shortest path is (0,0) -> (0,1) -> (0,2)
    # First step should be Direction.RIGHT
    assert bfs_first_step(start, target, obstacles) == Direction.RIGHT
    
    obstacles = {Point(0, 1)}
    # Path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2) or (0,0) -> (1,0) -> (1,1) -> (0,1) [obs]
    # Path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2). First step: DOWN
    assert bfs_first_step(start, target, obstacles) == Direction.DOWN
    
    obstacles = {Point(0, 1), Point(1, 0)}
    assert bfs_first_step(start, target, obstacles) is None

def test_flood_count():
    obstacles = {Point(1, 1)}
    # 3x3 board
    # (0,0) (0,1) (0,2)
    # (1,0) (1,1) (1,2)
    # (2,0) (2,1) (2,2)
    # start (0,0), obstacle (1,1)
    # Reachable: (0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)
    # Total 8
    assert flood_count(Point(0, 0), obstacles, 3, 3) == 8
    
    obstacles = {Point(0, 1), Point(1, 0)}
    # Reachable: only (0,0)
    assert flood_count(Point(0, 0), obstacles, 3, 3) == 1

def test_path_to_tail_direction():
    body = deque([Point(0, 0), Point(0, 1), Point(0, 2)])
    obstacles = set()
    # Head (0,0), Tail (0,2). First step should be RIGHT
    assert path_to_tail_direction(body, obstacles, 3, 3) == Direction.RIGHT
    
    obstacles = {Point(0, 1)}
    # Path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2)
    assert path_to_tail_direction(body, obstacles, 3, 3) == Direction.DOWN

def test_snake_init():
    # Start (0,0), Dir RIGHT, Len 3. Body should be (0,0), (0,-1), (0,-2)
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    assert list(s.body) == [Point(0, 0), Point(0, -1), Point(0, -2)]
    assert s.direction == Direction.RIGHT
    assert s.alive is True
    assert s.pending_grow == 0

def test_snake_properties():
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    assert s.head == Point(0, 0)
    assert s.body_set() == {Point(0, 0), Point(0, -1), Point(0, -2)}
    assert s.next_head(Direction.RIGHT) == Point(0, 1)

def test_snake_commit_move_no_grow():
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    # body: (0,0), (0,-1), (0,-2)
    s.commit_move(Point(0, 1), grow=False)
    # body: (0,1), (0,0), (0,-1)
    assert list(s.body) == [Point(0, 1), Point(0, 0), Point(0, -1)]
    assert len(s.body) == 3

def test_snake_commit_move_grow_true():
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    s.commit_move(Point(0, 1), grow=True)
    # body: (0,1), (0,0), (0,-1), (0,-2)
    assert list(s.body) == [Point(0, 1), Point(0, 0), Point(0, -1), Point(0, -2)]
    assert len(s.body) == 4

def test_snake_commit_move_pending_grow():
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    s.pending_grow = 1
    s.commit_move(Point(0, 1), grow=False)
    # body: (0,1), (0,0), (0,-1), (0,-2)
    assert list(s.body) == [Point(0, 1), Point(0, 0), Point(0, -1), Point(0, -2)]
    assert s.pending_grow == 0
    assert len(s.body) == 4

def test_snake_commit_move_both_grow():
    s = Snake(Point(0, 0), Direction.RIGHT, 3)
    s.pending_grow = 1
    s.commit_move(Point(0, 1), grow=True)
    # Both grow=True and pending_grow > 0. 
    # Most likely takes 1 growth and decrements pending_grow.
    assert len(s.body) == 4
    assert s.pending_grow == 0

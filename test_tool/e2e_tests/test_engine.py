import pytest
from snake_battle import Direction, Snake, GameState, GameEngine
from collections import deque

def test_direction_deltas():
    """Verify Direction enum deltas are correct."""
    assert Direction.UP.value == (-1, 0)
    assert Direction.DOWN.value == (1, 0)
    assert Direction.LEFT.value == (0, -1)
    assert Direction.RIGHT.value == (0, 1)

def test_snake_deep_copy():
    """Verify Snake.copy() is a true deep copy."""
    s = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    try:
        s_copy = s.copy()
    except AttributeError:
        pytest.fail("Snake.copy() method is missing")
    
    assert s_copy is not s
    assert s_copy.head == s.head
    
    # Move original and ensure copy doesn't change
    s.move(Direction.UP)
    assert s_copy.head != s.head, "Deep copy failed: changes to original affect copy"

def test_gamestate_occupied_set():
    """Verify GameState.occupied_set() with and without tail exclusion."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s1 = Snake((10, 10), Direction.LEFT, "Beta", 2)
    state = GameState(s0, s1, set(), set(), 30, 15, 0, False)
    
    try:
        occ_no_excl = state.occupied_set(exclude_tails=False)
        occ_excl = state.occupied_set(exclude_tails=True)
    except AttributeError:
        pytest.fail("GameState.occupied_set() method is missing")
    
    assert s0.head in occ_no_excl
    assert s0.tail in occ_no_excl
    assert s0.head in occ_excl
    assert s0.tail not in occ_excl

def test_engine_tick_wall_kill():
    """(a) wall kill when head goes out of bounds."""
    s0 = Snake((1, 5), Direction.UP, "Alpha", 1)
    s1 = Snake((10, 10), Direction.RIGHT, "Beta", 2)
    walls = {(0, 5)}
    state = GameState(s0, s1, set(), walls, 30, 15, 0, False)
    
    new_state = GameEngine.tick(state, Direction.UP, Direction.RIGHT)
    assert new_state.my_snake.alive is False, "Snake should die when hitting wall"

def test_engine_tick_self_collision():
    """(b) self-collision death."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    # Force body to be in the path of the next move
    s0.body = deque([(5, 5), (5, 6), (6, 6), (6, 5)])
    s1 = Snake((10, 10), Direction.RIGHT, "Beta", 2)
    state = GameState(s0, s1, set(), set(), 30, 15, 0, False)
    
    # Move RIGHT into (5, 6)
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.RIGHT)
    assert new_state.my_snake.alive is False, "Snake should die when hitting itself"

def test_engine_tick_truncation_kill():
    """(c) truncation kill when attacker head hits opponent body cell."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s1 = Snake((5, 8), Direction.RIGHT, "Beta", 2)
    # Beta body at (5, 7), (5, 6)
    s1.body = deque([(5, 8), (5, 7), (5, 6)])
    state = GameState(s0, s1, set(), set(), 30, 15, 0, False)
    
    # Alpha moves RIGHT into (5, 6)
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.RIGHT)
    assert new_state.my_snake.alive is False, "Snake should die when hitting opponent's body"

def test_engine_tick_head_to_head_longer_wins():
    """(d) head-to-head with longer snake winning."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s0.body = deque([(5, 5), (5, 4), (5, 3), (5, 2)]) # Len 4
    s1 = Snake((5, 7), Direction.LEFT, "Beta", 2)
    s1.body = deque([(5, 7), (5, 8), (5, 9)]) # Len 3
    state = GameState(s0, s1, set(), set(), 30, 15, 0, False)
    
    # Both move to (5, 6)
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.LEFT)
    assert new_state.my_snake.alive is True, "Longer snake should survive head-to-head"
    assert new_state.opponent.alive is False, "Shorter snake should die in head-to-head"

def test_engine_tick_head_to_head_equal_death():
    """(e) head-to-head equal length mutual death."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s0.body = deque([(5, 5), (5, 4), (5, 3)]) # Len 3
    s1 = Snake((5, 7), Direction.LEFT, "Beta", 2)
    s1.body = deque([(5, 7), (5, 8), (5, 9)]) # Len 3
    state = GameState(s0, s1, set(), set(), 30, 15, 0, False)
    
    # Both move to (5, 6)
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.LEFT)
    assert new_state.my_snake.alive is False, "Equal length snakes should both die"
    assert new_state.opponent.alive is False, "Equal length snakes should both die"

def test_engine_food_eating():
    """Verify food eating grows snake body by 1, just_ate flag set, food respawned."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s1 = Snake((10, 10), Direction.RIGHT, "Beta", 2)
    foods = {(5, 6)}
    state = GameState(s0, s1, foods, set(), 30, 15, 0, False)
    
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.RIGHT)
    
    assert new_state.my_snake.length == 4, "Snake should grow after eating food"
    assert getattr(new_state.my_snake, 'just_ate', False) is True, "just_ate flag should be set"
    assert (5, 6) not in new_state.foods, "Eaten food should be removed"
    assert len(new_state.foods) >= 5, "Food should be respawned"

def test_engine_tail_shrink():
    """Verify tail shrinks each tick when not just_ate (interpreted as hunger/shrink period)."""
    s0 = Snake((5, 5), Direction.RIGHT, "Alpha", 1)
    s1 = Snake((10, 10), Direction.RIGHT, "Beta", 2)
    # Set frame to shrink interval multiple and shrinking=True
    state = GameState(s0, s1, set(), set(), 30, 15, 30, True)
    
    initial_len = s0.length
    new_state = GameEngine.tick(state, Direction.RIGHT, Direction.RIGHT)
    
    # In hunger mode on tick 30, length should decrease
    assert new_state.my_snake.length < initial_len, "Snake should shrink during hunger phase"

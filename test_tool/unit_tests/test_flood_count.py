import pytest
from snake_battle import flood_count

def test_flood_count_isolated_region():
    """Verify flood_count correctly identifies isolated territory pockets."""
    # 3x3 grid, isolate (0,0) by blocking (0,1) and (1,0)
    obstacles = {(0, 1), (1, 0)}
    
    # Only (0,0) is reachable
    assert flood_count((0, 0), obstacles, 3, 3) == 1
    # The remaining 7 cells are connected
    assert flood_count((1, 1), obstacles, 3, 3) == 7

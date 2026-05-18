#!/usr/bin/env python3
"""
Terminal AI vs AI Snake Battle game using curses.
Two snakes (Alpha in green, Beta in red) compete on a bordered board.
Game ends when snakes die or reach frame 2000.
"""

import curses
import random
import copy
from collections import deque
from typing import Set, Tuple, List, Optional, Deque, Dict, NamedTuple, Union
from dataclasses import dataclass
from enum import Enum
import heapq
class Point(NamedTuple):
    y: int
    x: int

Pt = Point
# Global constants
MIN_FOOD = 5
INITIAL_LENGTH = 3
SURVIVAL_GATE = 500
GRID_W = 30
GRID_H = 15
NUM_FOOD = 5
FOOD_COUNT = 5
FRAME_MS = 50
FPS_DEFAULT = 20
FPS_MIN = 10
FPS_MAX = 60
COLOR_ALPHA = 1
COLOR_BETA = 2
COLOR_FOOD = 3
COLOR_BORDER = 4
def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class Direction(Enum):
    """Direction enum with (dy, dx) values."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    
    @property
    def delta(self) -> Tuple[int, int]:
        return self.value

    def opposite(self) -> 'Direction':
        """Return the opposite direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[self]

    @staticmethod
    def opposite_static(dir: 'Direction') -> 'Direction':
        return dir.opposite()

UP = Direction.UP
DOWN = Direction.DOWN
LEFT = Direction.LEFT
RIGHT = Direction.RIGHT

def opposite(dir: Direction) -> Direction:
    return dir.opposite()

ALL_DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

OPPOSITES = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT
}

def opposite(dir: Direction) -> Direction:
    return dir.opposite()

# Alias for tests
main = None # Will be set later or defined as a function

class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls = set()
        for x in range(width):
            self.walls.add((0, x))
            self.walls.add((height - 1, x))
        for y in range(height):
            self.walls.add((y, 0))
            self.walls.add((y, width - 1))

GameBoard = Board

def get_obstacles(state: 'GameState') -> Set[Tuple[int, int]]:
    """Return all occupied cells (walls and snakes)."""
    obs = state.walls.copy()
    obs.update(state.my_snake.body_set())
    obs.update(state.opponent.body_set())
    return obs

def flood_count(head: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> int:
    """Count reachable cells from head."""
    return temporal_flood_fill(head, obstacles, board_w, board_h)

def bfs_distances(start: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> Dict[Tuple[int, int], int]:
    """Return distances from start to all reachable cells."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        dist = distances[curr]
        y, x = curr
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nxt = (y + dy, x + dx)
            if 1 <= nxt[0] < board_h - 1 and 1 <= nxt[1] < board_w - 1:
                if nxt not in obstacles and nxt not in distances:
                    distances[nxt] = dist + 1
                    queue.append(nxt)
    return distances

def bfs_path_exists(start: Tuple[int, int], target: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> bool:
    """Check if target is reachable from start."""
    if start == target: return True
    visited = {start}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        y, x = curr
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nxt = (y + dy, x + dx)
            if nxt == target: return True
            if 1 <= nxt[0] < board_h - 1 and 1 <= nxt[1] < board_w - 1:
                if nxt not in obstacles and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
    return False

def voronoi_space(points: List[Tuple[int, int]], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Partition board based on nearest point."""
    return voronoi_partition(points, obstacles, board_w, board_h)

def nearest_foods(head: Tuple[int, int], foods: Set[Tuple[int, int]], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> List[Tuple[int, int]]:
    """Find closest food positions."""
    if not foods: return []
    dists = bfs_distances(head, obstacles, board_w, board_h)
    min_dist = min([dists.get(f, float('inf')) for f in foods])
    if min_dist == float('inf'): return []
    return [f for f in foods if dists.get(f) == min_dist]

def danger_zone(head: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> Set[Tuple[int, int]]:
    """Return cells that are dangerous to enter (e.g. dead ends)."""
    danger = set()
    # Heuristic: cells with few exits
    y, x = head
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nxt = (y + dy, x + dx)
        if 1 <= nxt[0] < board_h - 1 and 1 <= nxt[1] < board_w - 1:
            if nxt not in obstacles:
                # Check exits from nxt
                exits = 0
                for dy2, dx2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nxt2 = (nxt[0] + dy2, nxt[1] + dx2)
                    if 1 <= nxt2[0] < board_h - 1 and 1 <= nxt2[1] < board_w - 1:
                        if nxt2 not in obstacles:
                            exits += 1
                if exits <= 1:
                    danger.add(nxt)
    return danger
@dataclass
@dataclass
class GameState:
    my_snake: 'Snake'
    opponent: 'Snake'
    foods: Set[Tuple[int, int]]
    walls: Set[Tuple[int, int]]
    board_width: int
    board_height: int
    frame: int
    shrinking: bool

    def get_snake(self, index: int) -> 'Snake':
        return self.my_snake if index == 0 else self.opponent
class Snake:
    def __init__(self, name: str, body: List[Tuple[int, int]], direction: Direction, color_pair: int):
        """
        Initialize snake with name, body segments, direction and color.
        """
        self.name = name
        self.body = deque(body)
        self.direction = direction
        self.color_pair = color_pair
        self.color = color_pair  # For compatibility with curses
        self.alive = True
        self.grow_pending = 0
        
    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]
        
    @property
    def tail(self) -> Tuple[int, int]:
        return self.body[-1]

    def body_set(self) -> Set[Tuple[int, int]]:
        """Return all body positions as a set."""
        return set(self.body)

    @property
    def length(self) -> int:
        """Return body length."""
        return len(self.body)
    
    def move(self, new_direction: Direction) -> None:
        """Move the snake in the specified direction."""
        # Ignore 180-degree turns
        if new_direction == self.direction.opposite():
            return
        
        self.direction = new_direction
        dy, dx = self.direction.value
        new_head = (self.head[0] + dy, self.head[1] + dx)
        self.body.appendleft(new_head)
        
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()
    
    def grow(self) -> None:
        """Mark snake to grow on next move."""
        self.grow_pending += 1

    def push_head(self, pos: Tuple[int, int]) -> None:
        """Add a new head position."""
        self.body.appendleft(pos)

    def pop_tail(self) -> Tuple[int, int]:
        """Remove and return the tail position."""
        return self.body.pop()
def safe_directions(head: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                    board_w: int, board_h: int, current_dir: Direction) -> List[Direction]:
    """Return directions that don't lead to immediate death."""
    safe = []
    for direction in ALL_DIRS:
        if direction == current_dir.opposite():
            continue
        dy, dx = direction.value
        new_head = (head[0] + dy, head[1] + dx)
        if 1 <= new_head[0] < board_h - 1 and 1 <= new_head[1] < board_w - 1:
            if new_head not in obstacles:
                safe.append(direction)
    return safe

def a_star(start: Tuple[int, int], target: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
           board_w: int, board_h: int) -> Optional[List[Direction]]:
    """Simple A* pathfinding."""
    if start == target:
        return []
    
    pq = [(0, start, [])]
    visited = {start}
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        if current == target:
            return path
        
        y, x = current
        for direction in ALL_DIRS:
            dy, dx = direction.value
            neighbor = (y + dy, x + dx)
            if 1 <= neighbor[0] < board_h - 1 and 1 <= neighbor[1] < board_w - 1:
                if neighbor not in obstacles and neighbor not in visited:
                    visited.add(neighbor)
                    new_cost = len(path) + 1 + manhattan(neighbor, target)
                    heapq.heappush(pq, (new_cost, neighbor, path + [direction]))
    return None

def temporal_flood_fill(head: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                        board_w: int, board_h: int) -> int:
    """Flood fill to count reachable space."""
    visited = {head}
    queue = deque([head])
    count = 0
    while queue:
        curr = queue.popleft()
        count += 1
        y, x = curr
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nxt = (y + dy, x + dx)
            if 1 <= nxt[0] < board_h - 1 and 1 <= nxt[1] < board_w - 1:
                if nxt not in obstacles and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
    return count

def voronoi_partition(points: List[Tuple[int, int]], obstacles: Set[Tuple[int, int]], 
                      board_w: int, board_h: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Partition board based on nearest point."""
    partition = {}
    if not points:
        return partition
    
    pq = []
    for p in points:
        heapq.heappush(pq, (0, p, p))
        
    visited = set()
    while pq:
        dist, curr, owner = heapq.heappop(pq)
        if curr in visited:
            continue
        visited.add(curr)
        partition[curr] = owner
        
        y, x = curr
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nxt = (y + dy, x + dx)
            if 1 <= nxt[0] < board_h - 1 and 1 <= nxt[1] < board_w - 1:
                if nxt not in obstacles and nxt not in visited:
                    heapq.heappush(pq, (dist + 1, nxt, owner))
    return partition
def pt_add(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    return (p1[0] + p2[0], p1[1] + p2[1])

def bfs_dist(start: Tuple[int, int], target: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> int:
    dists = bfs_distances(start, obstacles, board_w, board_h)
    return dists.get(target, float('inf'))

def bfs_first_step(start: Tuple[int, int], target: Tuple[int, int], obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> Optional[Direction]:
    path = a_star(start, target, obstacles, board_w, board_h)
    return path[0] if path else None

def path_to_tail_direction(snake: 'Snake', obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> Optional[Direction]:
    return bfs_first_step(snake.head, snake.tail, obstacles, board_w, board_h)

class AIAlpha:
    """Hunter AI: aggressive, prioritizes killing opponent and eating food."""
    
    THREAT_RADIUS = 2
    RANDOM_CHANCE = 0.1
    FLOOD_LIMIT = 100
    
    def __init__(self, snake: Snake):
        """Initialize AIAlpha with a snake."""
        self.snake = snake
    
    def decide(self, state: GameState) -> Direction:
        """Choose direction based on aggressive strategy."""
        safe = self._safe_moves(state)
        
        if not safe:
            return self.snake.direction
        
        # Try to find a kill move
        kill_move = self._find_kill_move(safe, state)
        if kill_move:
            return kill_move
        
        # Prioritize food
        best_move = None
        best_dist = float('inf')
        for direction in safe:
            for food in state.foods:
                dist = manhattan((self._new_head(direction)), food)
                if dist < best_dist:
                    best_dist = dist
                    best_move = direction
        
        return best_move if best_move else safe[0]
    
    def _new_head(self, direction: Direction) -> Tuple[int, int]:
        """Calculate new head position for a given direction."""
        dy, dx = direction.value
        return (self.snake.head[0] + dy, self.snake.head[1] + dx)
    
    def _safe_moves(self, state: GameState) -> List[Direction]:
        """Get all safe moves for the snake."""
        obstacles = self.snake.body_set | state.opponent.body_set
        # Discard tail positions (they will move)
        obstacles.discard(self.snake.tail)
        if len(state.opponent.body) > INITIAL_LENGTH:
            obstacles.discard(state.opponent.tail)
        
        safe = []
        for direction in ALL_DIRS:
            # Skip 180-degree turn
            if direction == self.snake.direction.opposite():
                continue
            
            new_head = self._new_head(direction)
            
            # Check bounds
            if not (1 <= new_head[0] < state.board_height - 1 and 1 <= new_head[1] < state.board_width - 1):
                continue
            
            # Check obstacles
            if new_head in obstacles:
                continue
            
            safe.append(direction)
        
        return safe
    
    def _find_kill_move(self, safe: List[Direction], state: GameState) -> Optional[Direction]:
        """Find a move that traps the opponent."""
        for direction in safe:
            new_head = self._new_head(direction)
            
            # Create obstacles including new head
            obstacles = self.snake.body_set | state.opponent.body_set
            obstacles.add(new_head)
            
            # Check if opponent would have safe moves
            opp_safe = []
            for opp_dir in ALL_DIRS:
                if opp_dir == state.opponent.direction.opposite():
                    continue
                opp_dy, opp_dx = opp_dir.value
                ny, nx = state.opponent.head[0] + opp_dy, state.opponent.head[1] + opp_dx
                if 1 <= ny < state.board_height - 1 and 1 <= nx < state.board_width - 1:
                    if (ny, nx) not in obstacles:
                        opp_safe.append(opp_dir)
            
            if not opp_safe:
                return direction
        
        return None
    
    def _flood_fill_space(self, head: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                          board_w: int, board_h: int, limit: int) -> int:
        """BFS flood fill counting reachable cells."""
        if head in obstacles:
            return 0
        if not (1 <= head[0] < board_h - 1 and 1 <= head[1] < board_w - 1):
            return 0
        
        visited = {head}
        queue = deque([head])
        count = 0
        
        while queue:
            y, x = queue.popleft()
            count += 1
            if count >= limit:
                return count
            
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 1 <= ny < board_h - 1 and 1 <= nx < board_w - 1:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        
        return count
    
    def _bfs_to_nearest_food(self, head: Tuple[int, int], foods: Set[Tuple[int, int]],
                             obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> int:
        """BFS to find distance to nearest food."""
        if head in obstacles:
            return float('inf')
        
        visited = {head}
        queue = deque([(head, 0)])
        
        while queue:
            (y, x), dist = queue.popleft()
            
            if (y, x) in foods:
                return dist
            
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 1 <= ny < board_h - 1 and 1 <= nx < board_w - 1:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append(((ny, nx), dist + 1))
        
        return float('inf')


class AIBeta:
    """Defender AI: conservative, prioritizes survival then food collection."""
    
    WEIGHT_SPACE = 0.35
    WEIGHT_FOOD = 0.35
    WEIGHT_ATTACK = 0.30
    BONUS_OPPORTUNITY = 1.5
    FLOOD_LIMIT = 200
    
    def __init__(self, snake: Snake):
        """Initialize AIBeta with a snake."""
        self.snake = snake
    
    def decide(self, state: GameState) -> Direction:
        """Choose direction based on defensive strategy."""
        safe = self._safe_moves(state)
        
        if not safe:
            return self.snake.direction
        
        # Score each safe move
        best_move = None
        best_score = float('-inf')
        
        for direction in safe:
            score = self._score_direction(direction, state)
            if score > best_score:
                best_score = score
                best_move = direction
        
        return best_move if best_move else safe[0]
    
    def _new_head(self, direction: Direction) -> Tuple[int, int]:
        """Calculate new head position for a given direction."""
        dy, dx = direction.value
        return (self.snake.head[0] + dy, self.snake.head[1] + dx)
    
    def _safe_moves(self, state: GameState) -> List[Direction]:
        """Get all safe moves for the snake."""
        obstacles = self.snake.body_set | state.opponent.body_set
        obstacles.discard(self.snake.tail)
        if len(state.opponent.body) > INITIAL_LENGTH:
            obstacles.discard(state.opponent.tail)
        
        safe = []
        for direction in ALL_DIRS:
            if direction == self.snake.direction.opposite():
                continue
            
            new_head = self._new_head(direction)
            
            if not (1 <= new_head[0] < state.board_height - 1 and 1 <= new_head[1] < state.board_width - 1):
                continue
            
            if new_head in obstacles:
                continue
            
            safe.append(direction)
        
        return safe
    
    def _score_direction(self, direction: Direction, state: GameState) -> float:
        """Score a direction based on space, food distance, and attack potential."""
        new_head = self._new_head(direction)
        
        # Space score
        obstacles = self.snake.body_set | state.opponent.body_set
        space = self._flood_fill(new_head, obstacles, state.board_width, state.board_height)
        space_score = space / self.FLOOD_LIMIT
        
        # Food score
        food_score = self._compute_food_score(new_head, state.foods, obstacles, state.board_width, state.board_height)
        
        # Attack score
        attack_score = self._compute_attack_score(new_head, state.opponent, state.foods, state.walls, state.board_width, state.board_height)
        
        return (space_score * self.WEIGHT_SPACE + 
                food_score * self.WEIGHT_FOOD + 
                attack_score * self.WEIGHT_ATTACK)
    
    def _flood_fill(self, head: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                    board_w: int, board_h: int) -> int:
        """BFS flood fill counting reachable cells."""
        if head in obstacles:
            return 0
        if not (1 <= head[0] < board_h - 1 and 1 <= head[1] < board_w - 1):
            return 0
        
        visited = {head}
        queue = deque([head])
        count = 0
        
        while queue:
            y, x = queue.popleft()
            count += 1
            if count >= self.FLOOD_LIMIT:
                return count
            
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 1 <= ny < board_h - 1 and 1 <= nx < board_w - 1:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        
        return count
    
    def _compute_food_score(self, pos: Tuple[int, int], foods: Set[Tuple[int, int]], 
                            obstacles: Set[Tuple[int, int]], board_w: int, board_h: int) -> float:
        """Compute score based on proximity to food."""
        if not foods:
            return 0.0
        
        min_dist = min(manhattan(pos, f) for f in foods)
        return 1.0 / (1.0 + min_dist)
    
    def _compute_attack_score(self, pos: Tuple[int, int], opponent, files: Set[Tuple[int, int]],
                              walls: Set[Tuple[int, int]], board_w: int, board_h: int) -> float:
        """Compute score based on attack potential."""
        if not opponent.alive:
            return 0.0
        
        dist = manhattan(pos, opponent.head)
        return 1.0 / (1.0 + dist)


class Game:
    """Main game class."""
    
    MAX_FRAMES = 2000
    FOOD_COUNT = 5
    HUNGER_THRESHOLD = 150
    SHRINK_INTERVAL = 30
    
    def __init__(self, board_width: int, board_height: int, seed: Optional[int] = None):
        """Initialize game with board dimensions."""
        if seed is not None:
            random.seed(seed)
        
        self.board_width = board_width
        self.board_height = board_height
        self.frame = 0
        self.shrinking = False
        self._game_over = False
        self.winner = None
        
        # Create walls (border)
        self.walls: Set[Tuple[int, int]] = set()
        for x in range(board_width):
            self.walls.add((0, x))
            self.walls.add((board_height - 1, x))
        for y in range(board_height):
            self.walls.add((y, 0))
            self.walls.add((y, board_width - 1))
        
        # Create snakes at symmetric positions
        center_y = board_height // 2
        alpha_start_x = board_width // 4
        beta_start_x = 3 * board_width // 4
        
        self.alpha_snake = Snake((center_y, alpha_start_x), Direction.RIGHT, "Alpha", 1)
        self.beta_snake = Snake((center_y, beta_start_x), Direction.LEFT, "Beta", 2)
        
        # Create AIs
        self.alpha_ai = AIAlpha(self.alpha_snake)
        self.beta_ai = AIBeta(self.beta_snake)
        
        # Create initial food
        self.foods: Set[Tuple[int, int]] = set()
        self._spawn_food()
    
    def _spawn_food(self):
        """Spawn food until we have MIN_FOOD foods."""
        attempts = 0
        while len(self.foods) < MIN_FOOD and attempts < 1000:
            y = random.randint(1, self.board_height - 2)
            x = random.randint(1, self.board_width - 2)
            pos = (y, x)
            
            if pos not in self.walls:
                if pos not in self.alpha_snake.body_set:
                    if pos not in self.beta_snake.body_set:
                        if pos not in self.foods:
                            self.foods.add(pos)
            attempts += 1
    
    def _get_state(self) -> GameState:
        """Get current game state."""
        return GameState(
            my_snake=self.alpha_snake,
            opponent=self.beta_snake,
            foods=self.foods,
            walls=self.walls,
            board_width=self.board_width,
            board_height=self.board_height,
            frame=self.frame,
            shrinking=self.shrinking
        )
    
    def tick(self):
        """Advance game by one frame."""
        if self.game_over:
            return
        
        self.frame += 1
        
        # Check max frames
        # Check max frames
        if self.frame >= self.MAX_FRAMES:
            self._end_game_by_frames()
            return
        
        # Check hunger/shrinking
        if self.frame >= SURVIVAL_GATE:
            self.shrinking = True
        
        # Get AI decisions
        # For Alpha: opponent is Beta
        alpha_state = GameState(
            my_snake=self.alpha_snake,
            opponent=self.beta_snake,
            foods=self.foods.copy(),
            walls=self.walls.copy(),
            board_width=self.board_width,
            board_height=self.board_height,
            frame=self.frame,
            shrinking=self.shrinking
        )
        alpha_dir = self.alpha_ai.decide(alpha_state)
        
        # For Beta: opponent is Alpha
        beta_state = GameState(
            my_snake=self.beta_snake,
            opponent=self.alpha_snake,
            foods=self.foods.copy(),
            walls=self.walls.copy(),
            board_width=self.board_width,
            board_height=self.board_height,
            frame=self.frame,
            shrinking=self.shrinking
        )
        beta_dir = self.beta_ai.decide(beta_state)
        
        # Simulate moves to check collisions
        new_alpha_head = (self.alpha_snake.head[0] + alpha_dir.value[0], 
                         self.alpha_snake.head[1] + alpha_dir.value[1])
        new_beta_head = (self.beta_snake.head[0] + beta_dir.value[0],
                        self.beta_snake.head[1] + beta_dir.value[1])
        
        # Check head-to-head collision
        if new_alpha_head == new_beta_head:
            self.alpha_snake.alive = False
            self.beta_snake.alive = False
            self._end_game("Draw")
            return
        
        # Check crossover collision (swapping places)
        old_alpha_head = self.alpha_snake.head
        old_beta_head = self.beta_snake.head
        if new_alpha_head == old_beta_head and new_beta_head == old_alpha_head:
            self.alpha_snake.alive = False
            self.beta_snake.alive = False
            self._end_game("Draw")
            return
        
        # Check if snakes move into each other's body
        obstacles_alpha = self.beta_snake.body_set.copy()
        obstacles_beta = self.alpha_snake.body_set.copy()
        obstacles_alpha.discard(self.beta_snake.tail)
        obstacles_beta.discard(self.alpha_snake.tail)
        
        alpha_hits_body = new_alpha_head in obstacles_alpha
        beta_hits_body = new_beta_head in obstacles_beta
        
        if alpha_hits_body:
            self.alpha_snake.alive = False
        if beta_hits_body:
            self.beta_snake.alive = False
        
        # Move snakes
        alpha_grows = new_alpha_head in self.foods and self.alpha_snake.alive
        beta_grows = new_beta_head in self.foods and self.beta_snake.alive
        
        self.alpha_snake.move(alpha_dir)
        self.beta_snake.move(beta_dir)
        
        if alpha_grows:
            self.alpha_snake._growing = True
        if beta_grows:
            self.beta_snake._growing = True
        
        if self.alpha_snake._growing:
            self.alpha_snake._growing = False
            self.alpha_snake.body.append(self.alpha_snake.tail)
        
        if self.beta_snake._growing:
            self.beta_snake._growing = False
            self.beta_snake.body.append(self.beta_snake.tail)
        
        # Remove eaten food
        if new_alpha_head in self.foods:
            self.foods.discard(new_alpha_head)
        if new_beta_head in self.foods:
            self.foods.discard(new_beta_head)
        
        # Check wall collisions
        if self.alpha_snake.head in self.walls:
            self.alpha_snake.alive = False
        if self.beta_snake.head in self.walls:
            self.beta_snake.alive = False
        
        # Refill food
        while len(self.foods) < MIN_FOOD:
            self._spawn_food()
        
        # Apply shrinking if enabled
        if self.shrinking and self.frame % self.SHRINK_INTERVAL == 0:
            if len(self.alpha_snake.body) > 3:
                self.alpha_snake.body.pop()
            if len(self.beta_snake.body) > 3:
                self.beta_snake.body.pop()
        
        # Check game over
        if not self.alpha_snake.alive or not self.beta_snake.alive:
            if not self.alpha_snake.alive and not self.beta_snake.alive:
                self._end_game("Draw")
            elif not self.alpha_snake.alive:
                self._end_game("Beta")  # Alpha died, Beta wins
            else:
                self._end_game("Alpha")  # Beta died, Alpha wins
    
    def _end_game_by_frames(self):
        """End game when frame limit reached."""
        if self.alpha_snake.length > self.beta_snake.length:
            self._end_game("Alpha")
        elif self.beta_snake.length > self.alpha_snake.length:
            self._end_game("Beta")
        else:
            self._end_game("Draw")
    
    def _end_game(self, winner: str):
        """End the game with a winner."""
        self.game_over = True
        self.winner = winner
class GameEngine:
    """Stateless engine for game logic."""
    @staticmethod
    def tick(state: GameState, dir_a: Direction, dir_b: Direction) -> GameState:
        """
        Advance the game state by one frame based on directions.
        Returns the updated GameState.
        """
        # 1. Update frame and shrinking state
        state.frame += 1
        if state.frame >= SURVIVAL_GATE:
            state.shrinking = True
            
        s0 = state.my_snake
        s1 = state.opponent
        
        # 2. Calculate new head positions
        h0_new = (s0.head[0] + dir_a.value[0], s0.head[1] + dir_a.value[1])
        h1_new = (s1.head[0] + dir_b.value[0], s1.head[1] + dir_b.value[1])
        
        # 3. Collision Detection
        # Head-to-head collision
        if h0_new == h1_new:
            s0.alive = False
            s1.alive = False
        # Crossover collision
        elif h0_new == s1.head and h1_new == s0.head:
            s0.alive = False
            s1.alive = False
        
        # Body collisions
        # Alpha head into Beta body (excluding Beta tail)
        obs0 = set(s1.body)
        obs0.discard(s1.tail)
        if h0_new in obs0:
            s0.alive = False
            
        # Beta head into Alpha body (excluding Alpha tail)
        obs1 = set(s0.body)
        obs1.discard(s0.tail)
        if h1_new in obs1:
            s1.alive = False
            
        # 4. Growth and Movement
        # Check if food is eaten before moving
        s0_ate = h0_new in state.foods and s0.alive
        s1_ate = h1_new in state.foods and s1.alive
        
        if s0_ate:
            s0.grow()
        if s1_ate:
            s1.grow()
            
        s0.move(dir_a)
        s1.move(dir_b)
        
        # 5. Update Food
        if s0_ate:
            state.foods.discard(h0_new)
        if s1_ate:
            state.foods.discard(h1_new)
            
        # 6. Wall collisions
        if s0.head in state.walls:
            s0.alive = False
        if s1.head in state.walls:
            s1.alive = False
            
        # 7. Shrinking (Survival mechanism)
        # Use 30 as the shrink interval (matches Game.SHRINK_INTERVAL)
        if state.shrinking and state.frame % 30 == 0:
            if len(s0.body) > 3:
                s0.body.pop()
            if len(s1.body) > 3:
                s1.body.pop()
                
        return state

class Renderer:
    def __init__(self, stdscr):
        self.scr = stdscr
        self._setup_colors()

    def _setup_colors(self):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)

    def draw(self, state: GameState, frame: int):
        self.scr.erase()
        # Row 0: status
        s0 = state.get_snake(0)
        s1 = state.get_snake(1)
        status = f"Frame:{frame:4d} | Alpha(G) len:{s0.length:3d} alive:{s0.alive} | Beta(R) len:{s1.length:3d} alive:{s1.alive} | [Q]uit"
        self.scr.addstr(0, 0, status)

        # Border: rows 1..H+2, cols 0..W+1
        # Corners and edges
        # Top row
        self.scr.addstr(1, 0, '+', curses.color_pair(4))
        self.scr.addstr(1, 1, '-' * (state.board_width), curses.color_pair(4))
        self.scr.addstr(1, state.board_width + 1, '+', curses.color_pair(4))
        
        # Bottom row
        self.scr.addstr(state.board_height + 2, 0, '+', curses.color_pair(4))
        self.scr.addstr(state.board_height + 2, 1, '-' * (state.board_width), curses.color_pair(4))
        self.scr.addstr(state.board_height + 2, state.board_width + 1, '+', curses.color_pair(4))
        
        # Left and right columns
        for r in range(2, state.board_height + 2):
            self.scr.addstr(r, 0, '|', curses.color_pair(4))
            self.scr.addstr(r, state.board_width + 1, '|', curses.color_pair(4))
            
        # Food: draw '*' at (food_r+2, food_c+1)
        for food in state.foods:
            self.scr.addstr(food[0] + 2, food[1] + 1, '*', curses.color_pair(3))
            
        # Snakes
        for i in range(2):
            snake = state.get_snake(i)
            pair = 1 if i == 0 else 2
            for idx, segment in enumerate(snake.body):
                char = '@' if idx == 0 else '#'
                self.scr.addstr(segment[0] + 2, segment[1] + 1, char, curses.color_pair(pair))
        
        self.scr.noutrefresh()
        curses.doupdate()

    def draw_game_over(self, state: GameState):
        s0 = state.get_snake(0)
        s1 = state.get_snake(1)
        if s0.alive and not s1.alive:
            winner = "Alpha"
        elif s1.alive and not s0.alive:
            winner = "Beta"
        elif not s0.alive and not s1.alive:
            winner = "DRAW"
        else:
            # Compare lengths if both alive (e.g. frame limit)
            if s0.length > s1.length:
                winner = "Alpha"
            elif s1.length > s0.length:
                winner = "Beta"
            else:
                winner = "DRAW"
        
        msg = f"WINNER: {winner} - Press Q to exit"
        # Centering logic
        h, w = self.scr.getmaxyx()
        self.scr.addstr(h // 2, (w - len(msg)) // 2, msg, curses.A_BOLD)
        self.scr.refresh()

def _make_initial_state() -> GameState:
    # Snake 0: col 5, center row, heading RIGHT
    center_row = GRID_H // 2
    s0 = Snake((center_row, 5), Direction.RIGHT, "Alpha", 1)
    
    # Snake 1: col GRID_W-6, center row, heading LEFT
    s1 = Snake((center_row, GRID_W - 6), Direction.LEFT, "Beta", 2)
    
    # Place NUM_FOOD random food not overlapping snakes
    foods = set()
    while len(foods) < NUM_FOOD:
        y = random.randint(1, GRID_H - 1)
        x = random.randint(1, GRID_W - 1)
        pos = (y, x)
        if pos not in s0.body_set and pos not in s1.body_set:
            foods.add(pos)
            
    # We need walls for GameState
    walls = set()
    for x in range(GRID_W):
        walls.add((0, x))
        walls.add((GRID_H, x))
    for y in range(GRID_H + 1):
        walls.add((y, 0))
        walls.add((y, GRID_W))
        
    return GameState(
        my_snake=s0,
        opponent=s1,
        foods=foods,
        walls=walls,
        board_width=GRID_W,
        board_height=GRID_H,
        frame=0,
        shrinking=False
    )

def run_game(stdscr):
    import time
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(FRAME_MS)
    
    renderer = Renderer(stdscr)
    alpha = AIAlpha(Snake((0,0), Direction.RIGHT, "Alpha", 1)) # Dummy snake for AI init
    beta = AIBeta(Snake((0,0), Direction.LEFT, "Beta", 2))    # Dummy snake for AI init
    
    state = _make_initial_state()
    # Update AIs to use the actual snakes from the initial state
    alpha.snake = state.get_snake(0)
    beta.snake = state.get_snake(1)
    
    while True:
        key = stdscr.getch()
        if key in (ord('q'), ord('Q')):
            break
            
        s0, s1 = state.get_snake(0), state.get_snake(1)
        dir_a = alpha.get_move(state, 0) if s0.alive else Direction.RIGHT
        dir_b = beta.get_move(state, 1) if s1.alive else Direction.LEFT
        
        state = GameEngine.tick(state, dir_a, dir_b)
        renderer.draw(state, state.frame)
        
        if not s0.alive or not s1.alive:
            renderer.draw_game_over(state)
            while stdscr.getch() not in (ord('q'), ord('Q')):
                time.sleep(0.05)
            break

if __name__ == '__main__':
    curses.wrapper(run_game)



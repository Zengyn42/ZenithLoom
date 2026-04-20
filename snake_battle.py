#!/usr/bin/env python3
"""
Terminal AI vs AI Snake Battle game using curses.
Two snakes (Alpha in green, Beta in red) compete on a bordered board.
Game ends when snakes die or reach frame 2000.
"""

import curses
import random
from collections import deque
from typing import Set, Tuple, List, Optional, Deque, Dict, NamedTuple, Union
from dataclasses import dataclass
from enum import Enum
import heapq

class Pt(NamedTuple):
    y: int
    x: int

# Global constants
MIN_FOOD = 5
INITIAL_LENGTH = 3
SURVIVAL_GATE = 500

def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class Direction(Enum):
    """Direction enum with (dy, dx) values."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    
    def opposite(self) -> 'Direction':
        """Return the opposite direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[self]

ALL_DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

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

class Snake:
    def __init__(self, head: Tuple[int, int], direction: Direction, name: str, color_pair: int):
        """
        Initialize snake with head position and direction.
        """
        self.name = name
        self.direction = direction
        self.color_pair = color_pair
        self.color = color_pair  # For compatibility with curses
        self.alive = True
        self._growing = False
        self.body: Deque[Tuple[int, int]] = deque()
        
        # Initialize body with INITIAL_LENGTH segments
        # Head is at head position, body extends opposite to direction
        dy, dx = direction.value
        for i in range(INITIAL_LENGTH):
            self.body.append((head[0] - dy * i, head[1] - dx * i))
            
    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]
        
    @property
    def tail(self) -> Tuple[int, int]:
        return self.body[-1]

    @property
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
        
        if not self._growing:
            self.body.pop()
        else:
            self._growing = False
    
    def grow(self) -> None:
        """Mark snake to grow on next move."""
        self._growing = True

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
        self.game_over = False
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


class Renderer:
    """Renders the game using curses."""
    
    MIN_WIDTH = 40
    MIN_HEIGHT = 25
    
    def __init__(self, stdscr):
        """Initialize renderer with curses screen."""
        self.stdscr = stdscr
        curses.curs_set(0)
        self.stdscr.timeout(50)
        
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
    
    def render(self, game: Game):
        """Render the current game state."""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        for y in range(min(height - 1, game.board_height)):
            self.stdscr.move(4 + y, 1)
            for x in range(min(width, game.board_width)):
                pos = (y, x)
                
                if pos in game.alpha_snake.body_set:
                    if pos == game.alpha_snake.head:
                        self.stdscr.addstr(4 + y, x, '@', curses.color_pair(1) | curses.A_BOLD)
                    else:
                        self.stdscr.addstr(4 + y, x, '.', curses.color_pair(1))
                elif pos in game.beta_snake.body_set:
                    if pos == game.beta_snake.head:
                        self.stdscr.addstr(4 + y, x, '@', curses.color_pair(2) | curses.A_BOLD)
                    else:
                        self.stdscr.addstr(4 + y, x, '.', curses.color_pair(2))
                elif pos in game.foods:
                    self.stdscr.addstr(4 + y, x, '*', curses.color_pair(3))
                elif pos in game.walls:
                    self.stdscr.addstr(4 + y, x, '#')
                else:
                    self.stdscr.addstr(4 + y, x, ' ')
        
        status = f"Frame: {game.frame}/2000 | Alpha: {game.alpha_snake.name} | Beta: {game.beta_snake.name}"
        if game.game_over:
            status += f" | WINNER: {game.winner}"
        
        self.stdscr.addstr(0, 0, status[:width-1])
        self.stdscr.refresh()


def main(stdscr):
    """Main game loop."""
    game = Game(30, 15)
    renderer = Renderer(stdscr)
    
    while not game.game_over:
        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            return
        elif key == ord('r') or key == ord('R'):
            game = Game(30, 15)
        
        game.tick()
        renderer.render(game)


if __name__ == '__main__':
    curses.wrapper(main)

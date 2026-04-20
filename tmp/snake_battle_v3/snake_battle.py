#!/usr/bin/env python3
"""Snake Battle v3 - Two snake AI battle game."""

import random
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Set, Tuple, Optional, List


class Direction(Enum):
    """Cardinal directions for snake movement."""
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
            Direction.RIGHT: Direction.LEFT,
        }
        return opposites[self]

    @property
    def dy(self) -> int:
        """Return delta y for this direction."""
        return self.value[0]

    @property
    def dx(self) -> int:
        """Return delta x for this direction."""
        return self.value[1]


# Convenience aliases
UP = Direction.UP
DOWN = Direction.DOWN
LEFT = Direction.LEFT
RIGHT = Direction.RIGHT


@dataclass
class Snake:
    """Represents a snake in the game."""
    head: Tuple[int, int]
    direction: Direction
    name: str
    color_pair: int
    alive: bool = True
    body: deque = field(default_factory=lambda: deque())
    _growing: bool = False
    
    def __post_init__(self):
        """Initialize body with 3 segments."""
        self.body = deque()
        self.body.append((self.head[0], self.head[1]))
        # Add segments behind the head (opposite to direction)
        for i in range(1, 3):
            prev_y = self.head[0] - i * self.direction.dy
            prev_x = self.head[1] - i * self.direction.dx
            self.body.append((prev_y, prev_x))
    
    @property
    def body_set(self) -> Set[Tuple[int, int]]:
        """Return a set of all body positions for fast lookup."""
        return set(self.body)
    
    @property
    def length(self) -> int:
        """Return the snake's length."""
        return len(self.body)
    
    def move(self, new_direction: Direction):
        """Move the snake in the given direction."""
        # Ignore opposite direction (180 degree turn)
        if new_direction == self.direction.opposite():
            return
        
        self.direction = new_direction
        
        # Calculate new head position
        new_head = (
            self.body[0][0] + self.direction.dy,
            self.body[0][1] + self.direction.dx
        )
        
        # Add new head
        self.body.appendleft(new_head)
        
        # Remove tail unless growing
        if not self._growing:
            self.body.pop()
        else:
            self._growing = False
    
    def grow(self):
        """Mark the snake to grow on next move."""
        self._growing = True


@dataclass
class GameState:
    """Represents the complete game state."""
    my_snake: Snake
    opponent: Snake
    foods: Set[Tuple[int, int]]
    walls: Set[Tuple[int, int]]
    board_width: int
    board_height: int
    frame: int = 0
    shrinking: bool = False
    
    @property
    def interior(self) -> Set[Tuple[int, int]]:
        """All interior board positions (not walls)."""
        interior = set()
        for y in range(1, self.board_height - 1):
            for x in range(1, self.board_width - 1):
                interior.add((y, x))
        return interior


class AIAlpha:
    """Simple AI for snake named Alpha."""
    
    THREAT_RADIUS = 2
    RANDOM_CHANCE = 0.1
    
    def __init__(self, snake: Snake):
        self.snake = snake
    
    def decide(self, state: GameState) -> Direction:
        """Decide next move for the snake."""
        # Get all possible directions (not opposite)
        possible = [d for d in Direction if d != self.snake.direction.opposite()]
        
        if random.random() < self.RANDOM_CHANCE:
            return random.choice(possible)
        
        # Find nearest food
        foods = state.foods.copy() - self.snake.body_set - state.walls
        if not foods:
            return self.snake.direction
        
        nearest = None
        nearest_dist = float('inf')
        for food in foods:
            dist = abs(food[0] - self.snake.head[0]) + abs(food[1] - self.snake.head[1])
            if dist < nearest_dist:
                nearest = food
                nearest_dist = dist
        
        if nearest:
            # Move towards food
            dy = nearest[0] - self.snake.head[0]
            dx = nearest[1] - self.snake.head[1]
            
            if abs(dy) > abs(dx):
                preferred = Direction.DOWN if dy > 0 else Direction.UP
            else:
                preferred = Direction.RIGHT if dx > 0 else Direction.LEFT
            
            if preferred in possible:
                return preferred
        
        # Avoid opponent
        for d in possible:
            new_head = (self.snake.head[0] + d.dy, self.snake.head[1] + d.dx)
            if new_head not in state.opponent.body_set:
                return d
        
        return self.snake.direction
    
    def _flood_fill_space(self, start: Tuple[int, int], obstacles: Set[Tuple[int, int]], 
                         width: int, height: int, limit: int) -> int:
        """Flood fill to count reachable space."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue and len(visited) < limit:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        
        return len(visited)
    
    def _bfs_to_nearest_food(self, start: Tuple[int, int], foods: Set[Tuple[int, int]],
                             obstacles: Set[Tuple[int, int]], width: int, height: int) -> int:
        """BFS to find distance to nearest food."""
        if not foods:
            return float('inf')
        
        visited = {start: 0}
        queue = deque([start])
        
        while queue:
            y, x = queue.popleft()
            dist = visited[(y, x)]
            
            if (y, x) in foods:
                return dist
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited[(ny, nx)] = dist + 1
                        queue.append((ny, nx))
        
        return float('inf')


class AIBeta:
    """Advanced AI for snake named Beta."""
    
    WEIGHT_SPACE = 0.35
    WEIGHT_FOOD = 0.35
    WEIGHT_ATTACK = 0.30
    BONUS_OPPORTUNITY = 1.5
    FLOOD_LIMIT = 200
    
    def __init__(self, snake: Snake):
        self.snake = snake
    
    def decide(self, state: GameState) -> Direction:
        """Decide next move using weighted scoring."""
        possible = [d for d in Direction if d != self.snake.direction.opposite()]
        
        best_dir = self.snake.direction
        best_score = -float('inf')
        
        for d in possible:
            new_head = (self.snake.head[0] + d.dy, self.snake.head[1] + d.dx)
            
            # Skip if hitting wall or own body
            if (new_head in state.walls or 
                new_head in self.snake.body_set or
                new_head in state.opponent.body_set):
                continue
            
            score = 0
            
            # Space score
            obstacles = state.walls | self.snake.body_set | state.opponent.body_set
            space = self._flood_fill(new_head, obstacles, state.board_width, state.board_height)
            score += self.WEIGHT_SPACE * (space / self.FLOOD_LIMIT)
            
            # Food score
            food_score = self._compute_food_score(new_head, state.foods, obstacles,
                                                  state.board_width, state.board_height)
            score += self.WEIGHT_FOOD * food_score
            
            # Attack score
            attack_score = self._compute_attack_score(new_head, state.opponent,
                                                      state.foods, state.walls,
                                                      state.board_width, state.board_height)
            score += self.WEIGHT_ATTACK * attack_score
            
            if score > best_score:
                best_score = score
                best_dir = d
        
        return best_dir
    
    def _flood_fill(self, start: Tuple[int, int], obstacles: Set[Tuple[int, int]],
                    width: int, height: int) -> int:
        """Flood fill to count reachable space."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue and len(visited) < self.FLOOD_LIMIT:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (ny, nx) not in obstacles and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        
        return len(visited)
    
    def _compute_food_score(self, pos: Tuple[int, int], foods: Set[Tuple[int, int]],
                           obstacles: Set[Tuple[int, int]], width: int, height: int) -> float:
        """Compute score for food proximity."""
        if not foods:
            return 0.0
        
        distances = []
        for food in foods:
            dist = abs(food[0] - pos[0]) + abs(food[1] - pos[1])
            distances.append(dist)
        
        min_dist = min(distances) if distances else float('inf')
        # Normalize: closer = higher score
        return 1.0 / (1.0 + min_dist)
    
    def _compute_attack_score(self, pos: Tuple[int, int], opponent: Snake,
                             foods: Set[Tuple[int, int]], walls: Set[Tuple[int, int]],
                             width: int, height: int) -> float:
        """Compute score for attacking opponent."""
        # Score based on cornering opponent
        if not opponent.alive:
            return 0.0
        
        # Count opponent's escape options
        escape_options = 0
        opponent_head = opponent.head
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = opponent_head[0] + dy, opponent_head[1] + dx
            if 0 <= ny < height and 0 <= nx < width:
                if (ny, nx) not in walls and (ny, nx) not in opponent.body_set - {opponent_head}:
                    escape_options += 1
        
        # Fewer escape options = better attack position
        if escape_options <= 1:
            return 1.0
        elif escape_options <= 2:
            return 0.5
        return 0.0


class Game:
    """Main game logic class."""
    
    MAX_FRAMES = 2000
    FOOD_COUNT = 5
    HUNGER_THRESHOLD = 150
    SHRINK_INTERVAL = 30
    
    def __init__(self, width: int = 80, height: int = 50, seed: Optional[int] = None):
        self.board_width = width
        self.board_height = height
        self.rng = random.Random(seed)
        self.frame = 0
        self.shrinking = False
        self.game_over = False
        self.winner: Optional[str] = None
        
        # Initialize walls (border)
        self.walls = set()
        for x in range(width):
            self.walls.add((0, x))
            self.walls.add((height - 1, x))
        for y in range(height):
            self.walls.add((y, 0))
            self.walls.add((y, width - 1))
        
        # Initialize snakes
        mid_y = height // 2
        left_x = width // 4
        right_x = 3 * width // 4
        
        self.alpha_snake = Snake((mid_y, left_x), Direction.RIGHT, "Alpha", 1)
        self.beta_snake = Snake((mid_y, right_x), Direction.LEFT, "Beta", 2)
        
        # Initialize AIs
        self.alpha_ai = AIAlpha(self.alpha_snake)
        self.beta_ai = AIBeta(self.beta_snake)
        
        # Initialize foods
        self.foods = set()
        self._spawn_food()
    
    def _spawn_food(self):
        """Spawn random food in valid position."""
        while len(self.foods) < self.FOOD_COUNT:
            y = self.rng.randint(1, self.board_height - 2)
            x = self.rng.randint(1, self.board_width - 2)
            pos = (y, x)
            
            if (pos not in self.walls and
                pos not in self.alpha_snake.body_set and
                pos not in self.beta_snake.body_set):
                self.foods.add(pos)
    
    def tick(self):
        """Advance game by one frame."""
        self.frame += 1
        
        # Check max frames
        if self.frame >= self.MAX_FRAMES:
            self.game_over = True
            if self.alpha_snake.alive and self.beta_snake.alive:
                self.winner = "Draw"
            return
        
        # Create game state for AIs
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
        
        # Get AI decisions
        alpha_dir = self.alpha_ai.decide(alpha_state)
        beta_dir = self.beta_ai.decide(beta_state)
        
        # Calculate new positions
        old_alpha_head = self.alpha_snake.head
        old_beta_head = self.beta_snake.head
        
        new_alpha_head = (
            old_alpha_head[0] + alpha_dir.dy,
            old_alpha_head[1] + alpha_dir.dx
        )
        
        new_beta_head = (
            old_beta_head[0] + beta_dir.dy,
            old_beta_head[1] + beta_dir.dx
        )
        
        # Check head-to-head collision
        head_to_head = (new_alpha_head == new_beta_head and
                       self.alpha_snake.alive and
                       self.beta_snake.alive)
        
        if head_to_head:
            self.alpha_snake.alive = False
            self.beta_snake.alive = False
            self.game_over = True
            self.winner = "Draw"
            return
        
        # Check wall/body collisions (do BEFORE moving)
        alpha_hits = (new_alpha_head in self.walls or
                     new_alpha_head in self.beta_snake.body_set)
        beta_hits = (new_beta_head in self.walls or
                    new_beta_head in self.alpha_snake.body_set)
        
        if alpha_hits and self.alpha_snake.alive:
            self.alpha_snake.alive = False
        if beta_hits and self.beta_snake.alive:
            self.beta_snake.alive = False
        
        # Check if both dead
        if not self.alpha_snake.alive and not self.beta_snake.alive:
            self.game_over = True
            self.winner = "Draw"
            return
        
        # Check if one survived
        if self.game_over and self.winner is None:
            if self.alpha_snake.alive:
                self.winner = "Alpha"
            elif self.beta_snake.alive:
                self.winner = "Beta"
            else:
                self.winner = "Draw"
        
        if self.game_over:
            return
        
        # Update snake directions
        self.alpha_snake.direction = alpha_dir
        self.beta_snake.direction = beta_dir
        
        # Move snakes
        self.alpha_snake.move(self.alpha_snake.direction)
        self.beta_snake.move(self.beta_snake.direction)
        
        # Check food for alpha (after move)
        if self.alpha_snake.head in self.foods:
            self.foods.discard(self.alpha_snake.head)
            self.alpha_snake.grow()
        
        # Check food for beta (after move)
        if self.beta_snake.head in self.foods:
            self.foods.discard(self.beta_snake.head)
            self.beta_snake.grow()
        
        # Respawn food if needed
        self._spawn_food()
        
        # Check hunger/shrinking
        if self.frame % self.SHRINK_INTERVAL == 0:
            if self.alpha_snake.length >= 3:
                self.alpha_snake.body.pop()
            if self.beta_snake.length >= 3:
                self.beta_snake.body.pop()
        
        # Check if snake died from shrinking
        if not self.alpha_snake.alive or not self.beta_snake.alive:
            self.game_over = True
            if self.alpha_snake.alive:
                self.winner = "Alpha"
            elif self.beta_snake.alive:
                self.winner = "Beta"
            else:
                self.winner = "Draw"


class Renderer:
    """Handle curses-based rendering."""
    
    MIN_WIDTH = 40
    MIN_HEIGHT = 25
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(50)  # 20 FPS
        
        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Alpha (green)
        curses.init_pair(2, curses.COLOR_RED, -1)     # Beta (red)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Food (yellow)
        curses.init_pair(4, curses.COLOR_WHITE, -1)   # Walls (white)
    
    def render(self, game: Game):
        """Render the game state."""
        self.stdscr.clear()
        
        height, width = self.stdscr.getmaxyx()
        
        # Calculate scaling
        scale_x = width / game.board_width
        scale_y = height / game.board_height
        scale = min(scale_x, scale_y)
        
        # Center the game
        start_x = (width - game.board_width * scale) // 2
        start_y = (height - game.board_height * scale) // 2
        
        if start_y < 1:
            start_y = 1
        
        # Draw title
        title = f"Snake Battle v3 - Alpha vs Beta"
        try:
            self.stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        except curses.error:
            pass
        
        # Draw game info
        info = f"Frame: {game.frame} | Alpha: {game.alpha_snake.length} | Beta: {game.beta_snake.length} | Food: {len(game.foods)}"
        try:
            self.stdscr.addstr(1, (width - len(info)) // 2, info)
        except curses.error:
            pass
        
        # Draw game over screen
        if game.game_over:
            end_msg = f"GAME OVER - Winner: {game.winner}" if game.winner else "GAME OVER - Draw"
            try:
                self.stdscr.addstr(height // 2, (width - len(end_msg)) // 2, end_msg, 
                                  curses.A_BOLD | curses.A_REVERSE)
            except curses.error:
                pass
        
        # Draw walls
        for y, x in game.walls:
            try:
                cy, cx = int(start_y + y * scale), int(start_x + x * scale)
                self.stdscr.addstr(cy, cx, '#', curses.color_pair(4))
            except curses.error:
                pass
        
        # Draw food
        for y, x in game.foods:
            try:
                cy, cx = int(start_y + y * scale), int(start_x + x * scale)
                self.stdscr.addstr(cy, cx, '*', curses.color_pair(3))
            except curses.error:
                pass
        
        # Draw alpha snake
        if game.alpha_snake.alive:
            for i, (y, x) in enumerate(game.alpha_snake.body):
                try:
                    cy, cx = int(start_y + y * scale), int(start_x + x * scale)
                    ch = '@' if i == 0 else 'o'
                    self.stdscr.addstr(cy, cx, ch, curses.color_pair(1))
                except curses.error:
                    pass
        
        # Draw beta snake
        if game.beta_snake.alive:
            for i, (y, x) in enumerate(game.beta_snake.body):
                try:
                    cy, cx = int(start_y + y * scale), int(start_x + x * scale)
                    ch = '#' if i == 0 else 'o'
                    self.stdscr.addstr(cy, cx, ch, curses.color_pair(2))
                except curses.error:
                    pass
        
        self.stdscr.refresh()


def main():
    """Main entry point for curses-based UI."""
    def wrapper(stdscr):
        game = Game(80, 50, seed=42)
        renderer = Renderer(stdscr)
        
        while not game.game_over:
            renderer.render(game)
            key = stdscr.getch()
            if key == ord('q'):
                break
            game.tick()
        
        renderer.render(game)
        # Wait for key press
        while True:
            key = stdscr.getch()
            if key == ord('q') or key == ord(' '):
                break
    
    curses.wrapper(wrapper)


if __name__ == '__main__':
    main()

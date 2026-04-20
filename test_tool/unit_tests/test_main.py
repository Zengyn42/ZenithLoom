import pytest
from unittest.mock import MagicMock, patch
import curses
import time

# Import the main function from the file. 
# Since snake_battle.py has if __name__ == "__main__": curses.wrapper(main), 
# importing it shouldn't trigger the wrapper.
from snake_battle import main

def test_main_logic():
    # Mock stdscr
    stdscr = MagicMock()
    
    # Mock curses methods
    with patch('curses.curs_set') as mock_curs_set:
        # Setup getch to return 'q' immediately to exit the loop
        # First call in run_game loop -> 'q'
        # This should make run_game return 'quit', and then main should break.
        stdscr.getch.return_value = ord('q')
        
        # We also need to mock the other classes used inside main to avoid 
        # performing full simulations and curses rendering.
        with patch('snake_battle.GameEngine') as MockEngine, \
             patch('snake_battle.AIAlpha') as MockAIAlpha, \
             patch('snake_battle.AIBeta') as MockAIBeta, \
             patch('snake_battle.CursesUI') as MockUI:
            
            # Setup the engine mock to run for 0 iterations (or 1)
            # engine.running should be True then False, or we just rely on getch('q')
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.running = True 
            # Since we return 'quit' from run_game via getch('q'), we don't need to worry about infinite loop
            
            main(stdscr)
            
            # Verify cursor is hidden
            mock_curs_set.assert_called_once_with(0)
            # Verify non-blocking input is set
            stdscr.nodelay.assert_called_with(True)
            stdscr.timeout.assert_called_with(0)
            # Verify it tried to get a key
            stdscr.getch.assert_called()

def test_replay_logic():
    stdscr = MagicMock()
    
    with patch('curses.curs_set'), \
         patch('snake_battle.GameEngine') as MockEngine, \
         patch('snake_battle.AIAlpha'), \
         patch('snake_battle.AIBeta'), \
         patch('snake_battle.CursesUI') as MockUI:
        
        mock_engine_instance = MockEngine.return_value
        # We want to simulate: 
        # 1. Game runs until finished (running becomes False)
        # 2. Player presses 'r' to replay
        # 3. Player presses 'q' to quit
        
        # Case 1: Game loop ends naturally.
        # The first call to run_game:
        #   while engine.running: 
        #     key = stdscr.getch()
        #     ...
        #   ui.show_result()
        #   while True:
        #     key = stdscr.getch()
        #     if 'r' return 'replay'
        #     if 'q' return 'quit'
        
        mock_engine_instance.running = False # Ends first game immediately
        
        # getch sequence: 
        # 1. First call in run_game loop: getch() - engine.running is already False, so loop skipped.
        # 2. In wait for input loop: getch() returns 'r' -> returns 'replay'
        # 3. In second run_game loop: getch() - engine.running still False, loop skipped.
        # 4. In second wait for input loop: getch() returns 'q' -> returns 'quit'
        
        stdscr.getch.side_effect = [ord('r'), ord('q')]
        
        main(stdscr)
        
        # If it reached this point, it should have called run_game twice
        assert MockEngine.call_count == 2

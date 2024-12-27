# Feel free to modify the code to facilitate your implementation, e.g., add or modify functions, import other modules.
import argparse
import numpy as np
import copy

class Puzzle(object):
    def __init__(self, file_path=None, puzzle=None):
        self.size = 3
        self.goal_puzzle = np.array([[1, 2, 3],[8, 0, 4],[7, 6, 5]])

        if puzzle is not None:
            self.puzzle = puzzle
        elif file_path is not None:
            self.puzzle = self.read_puzzle(file_path)
            assert self.is_solvable(self.puzzle), '8-puzzle has an unsolvable initial state.'
        else:
            self.puzzle = self.make_puzzle()
            
    def make_puzzle(self):
        tiles = range(self.size ** 2)
        tiles = list(tiles)
        np.random.shuffle(tiles)

        while not self.is_solvable(np.array(tiles).reshape((self.size, self.size))):
            np.random.shuffle(tiles)

        return np.array(tiles).reshape((self.size, self.size))
    
    def read_puzzle(self, file_path):
        with open(file_path, 'r') as file:
            puzzle = np.array([list(map(int, line.strip().split())) for line in file.readlines()])
            assert puzzle.shape[0]==self.size and puzzle.shape[1]==self.size, "8-puzzle should have a 3 * 3 board."
        return puzzle

    def is_solvable(self, puzzle):
        # Based on http://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable
        goal_puzzle = self.goal_puzzle.flatten()
        goal_inversions = 0
        for i in range(len(goal_puzzle)):
            for j in range(i+1, len(goal_puzzle)):
                if goal_puzzle[i] > goal_puzzle[j] and goal_puzzle[i] != 0 and goal_puzzle[j] != 0:
                    goal_inversions += 1

        puzzle = puzzle.flatten()
        inversions = 0
        for i in range(len(puzzle)):
            for j in range(i+1, len(puzzle)):
                if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                    inversions += 1

        return inversions % 2 == goal_inversions % 2

    def misplaced_tiles(self):
        # Implement it if this heuristic is admissible
        h = 0
        new_puzzle = self.puzzle.flatten()
        goal_puzzle = self.goal_puzzle.flatten()
        for i in range(9):
            if ((new_puzzle[i] != goal_puzzle[i]) and new_puzzle[i] != 0):
                h += 1
        return h

    def manhattan_distance(self):
        # Implement it if this heuristic is admissible
        h = 0
        new_puzzle = list(self.puzzle.flatten())
        goal_puzzle = list(self.goal_puzzle.flatten())
        for value in new_puzzle:
            if (value != 0):
                i = new_puzzle.index(value)
                j = goal_puzzle.index(value)
                i_row, i_col = i // 3, i % 3
                j_row, j_col = j // 3, j % 3
                h += (abs(i_row - j_row) + abs(i_col - j_col))
        return h
    
    def nilsson_heuristic(self):
        # Implement it if this heuristic is admissible
        Manhattn_dist = self.manhattan_distance()
        Sequence_score = 0
        for i in range(self.puzzle.shape[0]):
            for j in range (self.puzzle.shape[1]):
                # Loop
                if (self.puzzle[i, j] != 0):
                    if ((i == self.puzzle.shape[0]) / 2 and (j == self.puzzle.shape[1])):
                        if (self.puzzle[i, j] != (i * self.puzzle.shape[0] + self.puzzle.shape[1])):
                            Sequence_score += 1
                    else:
                        if (self.puzzle[i, j] != (i * self.puzzle.shape[0] + self.puzzle.shape[1])):
                            Sequence_score += 2
        h = Manhattn_dist + 3 * Sequence_score
        return h
    
    def mostow_prieditis_heuristic(self):
        # Implement it if this heuristic is admissible
        h = 0
        new_puzzle = list(self.puzzle.flatten())
        goal_puzzle = list(self.goal_puzzle.flatten())
        for value in new_puzzle:
            if (value != 0):
                i = new_puzzle.index(value)
                j = goal_puzzle.index(value)
                i_row, i_col = i // 3, i % 3
                j_row, j_col = j // 3, j % 3
                if (i_row != j_row):
                    h += 1
                if (i_col != j_col):
                    h += 1
        return h
    

def a_star_algorithm(start_puzzle, heuristic):
    # Implement A* search here 
    num_node_expand = 0
    path = []
    frontier = {(heuristic(start_puzzle), start_puzzle): [start_puzzle]}      # dictionary, 一种puzzle状态对应一种path
    explored = []
    while frontier:
        # sort the dictionary
        sorted_frontier_dict = dict(sorted(frontier.items(), key=lambda item: item[0][0])) # Sort frontier to find the state with the lowest f value
        (current_f, current_Puzzle) = next(iter(sorted_frontier_dict))
        current_path = sorted_frontier_dict.pop((current_f, current_Puzzle))
        # check if current explored?
        current_Puzzle_tuple = tuple(map(tuple, current_Puzzle.puzzle))
        frontier.pop((current_f, current_Puzzle))

        if current_Puzzle_tuple in explored:
            continue
        explored.append(current_Puzzle_tuple)         # Add to explored array [Puzzle]

        num_node_expand += 1  # Increment the number of nodes expanded
        
        if np.array_equal(current_Puzzle.puzzle, start_puzzle.goal_puzzle):
            return [current_path, num_node_expand]
        
        
        empty_pair = np.argwhere(current_Puzzle.puzzle == 0).flatten()
        empty_x = empty_pair[0]
        empty_y = empty_pair[1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for d in directions:
            new_x, new_y = empty_x + d[0], empty_y + d[1]
            if (0 <= new_x < current_Puzzle.size and 0 <= new_y < current_Puzzle.size):
                new_Puzzle = copy.deepcopy(current_Puzzle)
                switch_num = current_Puzzle.puzzle[new_x, new_y]
                new_Puzzle.puzzle[empty_x, empty_y] = switch_num
                new_Puzzle.puzzle[new_x, new_y] = 0
                new_Puzzle_tuple = tuple(map(tuple, new_Puzzle.puzzle))
                if new_Puzzle_tuple not in explored:
                    new_f = heuristic(new_Puzzle) + len(current_path)  # Total cost
                    new_path = copy.deepcopy(current_path)
                    new_path.append(new_Puzzle)
                    frontier.update({(new_f, new_Puzzle) : new_path})
    
    return [path, num_node_expand] 
    '''num_node_expand = 0
    path = []

    frontier = {(heuristic(start_puzzle) + 0, start_puzzle): [start_puzzle]}      # dictionary, 一种puzzle状态对应一种path
    explored = []
    while frontier:
        sorted_frontier_list = sorted(frontier.items(), key=lambda item: item[0][0]) # Sort frontier to find the state with the lowest f value
        sorted_frontier_dict = dict(sorted_frontier_list)
        (current_f, current_Puzzle) = next(iter(sorted_frontier_dict))
        current_path = sorted_frontier_dict.pop((current_f, current_Puzzle))
        
        explored.append(current_Puzzle)         # Add to explored array [Puzzle]
        num_node_expand += 1  # Increment the number of nodes expanded
        
        if np.array_equal(current_Puzzle.puzzle, start_puzzle.goal_puzzle):
            path = frontier.get((current_f, current_Puzzle))
            break
        
        frontier.pop((current_f, current_Puzzle))
        
        puzzle_list = []
        
        empty_pair = np.argwhere(current_Puzzle.puzzle == 0).flatten()
        empty_x = empty_pair[0]
        empty_y = empty_pair[1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for d in directions:
            new_x, new_y = empty_x + d[0], empty_y + d[1]
            if (0 <= new_x < current_Puzzle.size and 0 <= new_y < current_Puzzle.size):
                new_Puzzle = copy.deepcopy(current_Puzzle)
                switch_num = current_Puzzle.puzzle[new_x, new_y]
                new_Puzzle.puzzle[empty_x, empty_y] = switch_num
                new_Puzzle.puzzle[new_x, new_y] = 0
                puzzle_list.append(new_Puzzle)
        
        for new_Puzzle in puzzle_list:
            if new_Puzzle not in explored:
                new_f = heuristic(new_Puzzle) + len(current_path)  # Total cost
                new_path = copy.deepcopy(current_path)
                new_path.append(new_Puzzle)
                frontier.update({(new_f, new_Puzzle) : new_path})
        
    return [path, num_node_expand]'''
    
    

def write_output(name, data, student_id):
    with open(name, 'w') as file:
        file.write(str(student_id) + '\n')
        for state in data:
            for row in state.puzzle:
                file.write(' '.join(map(str, row)) + '\n')
            file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--puzzle", default="puzzle_4.txt", help="Path to txt file containing 8-puzzle")
    parser.add_argument("-H", "--heuristic", type=int, help="Heuristic mode to use. 1: Use Misplaced Tiles; 2: Use Manhattan distance; 3: Use Nilsson Heuristic; 4: Use Mostow and Prieditis Heuristic", default=1, choices=[1, 2, 3, 4]) # You can change the allowable values to only those representing admissible heuristics
    parser.add_argument("-o", "--output_file", default="output_1.txt", help="Path to output txt file")

    args = parser.parse_args()

    if args.puzzle:
        initial_state = Puzzle(args.puzzle)
    else:
        initial_state = Puzzle()

    heuristic_idx = {
    1: "Misplaced Tiles",
    2: "Manhattan Distance",
    3: "Nilsson Heuristic",
    4: "Mostow and Prieditis Heuristic",
    }

    heuristics = {
    "Misplaced Tiles": lambda state: state.misplaced_tiles(),
    "Manhattan Distance": lambda state: state.manhattan_distance(),
    "Nilsson Heuristic": lambda state: state.nilsson_heuristic(),
    "Mostow and Prieditis Heuristic": lambda state: state.mostow_prieditis_heuristic(),
    }
    
    name = heuristic_idx[args.heuristic]
    heuristic = heuristics[name]
    print(f"Using {name}:")
    result_list = a_star_algorithm(initial_state, heuristic)
    if result_list:
        path, num_node_expand = result_list
        print(f"Solution found with {len(path) - 1} moves. {num_node_expand} nodes are expanded.")
        write_output(args.output_file, path, "20945880")
    else:
        print("No solution found.")
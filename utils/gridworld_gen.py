'''
Generator for multiple gridworlds and expert solution

grid states:
    0 = empty
    1 = agent
    2 = goal
    3 = agent reached goal
'''
import numpy as np
import pickle
from numpy.random import randint, sample, choice
OUTPUT_FILE = 'data/gridworld.dat'

WORLD_COUNT = 10 # How many worlds

# If true, grid size will be sampled from mean and variance, otherwise variance is 0
RANDOM_GRID_SIZE = True
GRID_SIZE_MEAN = 30
GRID_SIZE_VARIANCE = 9

# If true, agent will always start at the near center of the map
RANDOM_STARTING_POINT = True

# If true, required actions are sampled with mean an variance below, otherwise variance is 0
RANDOM_ACTION_COUNT = True
ACTION_COUNT_MEAN = 13
ACTION_COUNT_VARIANCE = 6

# If True, complex actions like left-left-up-up will be required.
SIMPLE_ACTIONS = True


def main():
    print('yo')
    #for i in range(WORLD_COUNT)
    worlds = []
    for i in range(WORLD_COUNT):
        world = generate()
        solver(world)
        worlds.append(world)

    export_gridworld(worlds)

    worlds = import_griddworlds()
    print('hello world')



def move(grid, dir, step, ):
    '''
    Moves agent in the direction desired, will insert into the next item on the grid
    :param grid: Entire grid world Step x W x H
    :param dir: Direction of movement
    :param step: *Current* step, will return the new step
    :return:
    '''

    if dir == 'up':
        delta = [-1, 0]
    elif dir == 'down':
        delta = [1, 0]
    elif dir == 'left':
        delta = [0, -1]
    elif dir == 'right':
        delta = [0, 1]

    agent = np.argwhere(grid[step] == 1)[0]
    goal = np.argwhere(grid[step] == 2)[0]

    new_pos = agent + delta
    step += 1

    if new_pos[0] == goal[0] and new_pos[1] == goal[1]: # Win
        grid[step, new_pos[0], new_pos[1]] = 3
    else:
        grid[step, new_pos[0], new_pos[1]] = 1
        grid[step, goal[0], goal[1]] = 2

    return grid, step

def generate():
    '''
    Generator function that generates the grid world matrix and space for all steps nessasary
    :return: Generated grid matrix
    '''
    # TODO Finish these conditions
    if not SIMPLE_ACTIONS:
        raise NotImplementedError


    # Grid
    if RANDOM_GRID_SIZE:
        gridsize = randint(GRID_SIZE_MEAN - GRID_SIZE_VARIANCE,
                           GRID_SIZE_MEAN + GRID_SIZE_VARIANCE+1)
    else:
        gridsize = GRID_SIZE_MEAN

    # Action count
    if RANDOM_ACTION_COUNT:
        actions_count = randint(ACTION_COUNT_MEAN - ACTION_COUNT_VARIANCE,
                                ACTION_COUNT_MEAN + ACTION_COUNT_VARIANCE + 1)
    else:
        actions_count = ACTION_COUNT_MEAN

    grid = np.zeros([actions_count + 1, gridsize, gridsize, ])  # Grid

    # Randomly a good starting point & goal
    for i in range(100):
        # Starting point
        starting_point = randint(gridsize, size=2) if RANDOM_STARTING_POINT else [(gridsize / 2), (gridsize / 2)]

        # Create goal
        if SIMPLE_ACTIONS:
            # Randomly assign actions
            goal = np.copy(starting_point)
            ridx = randint(2) # Randomly select axis
            sign = choice([-1, 1])
            goal[ridx] = goal[ridx] + sign * actions_count
            #goal[ridx] = 9
        else:
            pass
            # TODO implemented this

        if goal[0] >= 0 and goal[1] >=0 and goal[0] < gridsize and goal[1] < gridsize:
            break
        elif i == 99:
            raise BrokenPipeError('Could not find a good starting point')

    grid[0, starting_point[0], starting_point[1]] = 1
    grid[0, goal[0], goal[1]] = 2

    return grid

def solver(gridworld):
    agent = np.argwhere(gridworld[0] == 1)[0]
    goal = np.argwhere(gridworld[0] == 2)[0]

    x = goal[1] - agent[1]
    y = goal[0] - agent[0]

    step = 0
    if x > 0:
        for i in range(x):
            move(gridworld, 'right', i)
    else:
        x = np.abs(x) # Make positive
        for i in range(x):
            move(gridworld, 'left', i)

    if y > 0:
        for i in range(y):
            move(gridworld, 'down', i + x)
    else:
        y = np.abs(y) # make positive
        for i in range(y):
            move(gridworld, 'up', i + x)

def import_griddworlds(filename=OUTPUT_FILE):

    with open(filename, 'rb') as f:
        loaded_games = pickle.load(f)
    gridworlds = []
    for world in loaded_games:
        gridworlds.append(pickle.loads(world))
    return gridworlds

def export_gridworld(gridworlds, filename=OUTPUT_FILE):
    '''
    Exports generated gridworlds and write to the file specified above
    '''
    picked_worlds = []
    for gridworld in gridworlds:
        picked_worlds.append(gridworld.dumps())

    with open(filename, 'wb') as f:
        pickle.dump(picked_worlds, f)



def visualize(gridworld):
    '''
    Prints out a single gridworld state in the console. Cowboy is agent, flag is goal, and greenheart (if appears)
    means the agent has won. Game over
    '''
    # print upper boarder
    print(''.join(["==" for _ in range(gridworld.shape[1]+1)]))

    for row in gridworld:
        print('|', end='')
        for ele in row:
            if ele == 1:
                print('🤠', end='')
            elif ele == 2:
                print('🚩', end='')
            elif ele == 3:
                print('💚', end='')
            else:
                print('  ', end='')
        print('|')

    print(''.join(["==" for _ in range(gridworld.shape[1]+1)]))

    pass

if __name__ == '__main__':
    main()
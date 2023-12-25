from secrets import choice
import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

from itertools import combinations


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # for each timestep, check if there is a collision
    for t in range(max(len(path1), len(path2))):    
        # get the location of each agent at timestep t
        loc1 = get_location(path1, t)
        loc2 = get_location(path2, t)
        # check if there is a vertex collision
        if loc1 == loc2:
            return {'loc': [loc1], 'timestep': t}
        # check if there is an edge collision
        if t > 0:
            prev_loc1 = get_location(path1, t-1)
            prev_loc2 = get_location(path2, t-1)
            if loc1 == prev_loc2 and loc2 == prev_loc1:
                return {'loc': [loc1, loc2], 'timestep': t}
    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    # split the list into two list of paths
    collisions = []
    # for each pair of paths, detect collision
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            # for each pair of paths, call detect_collision
            collision = detect_collision(paths[i], paths[j])
            if collision is not None:
                collisions.append({'a1':i, 'a2':j, 'loc': collision['loc'], 'timestep': collision['timestep']})

    # return the list of collisions
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    # get the id of the two robots
    a1 = collision['a1']
    a2 = collision['a2']
    # get the location of the collision
    loc = collision['loc']
    # get the timestep of the collision
    t = collision['timestep']
    # create the two constraints
    if len(loc) == 1:
        # vertex collision
        constraint1 = {'agent': a1, 'loc': loc, 'timestep': t}
        constraint2 = {'agent': a2, 'loc': loc, 'timestep': t}        
    else:
        # edge collision
        constraint1 = {'agent': a1, 'loc': [loc[1],loc[0]], 'timestep': t}
        constraint2 = {'agent': a2, 'loc': loc, 'timestep': t}
        
    # return the two constraints
    return [constraint1, constraint2]


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    # get the id of the two robots
    a1 = collision['a1']
    a2 = collision['a2']
    # save the id of the agent that will be enforced
    agents = [a1, a2]
    # generate a random number to choose the agent
    choice = random.randint(0,1)
    # choose the agent
    agent = agents[choice]
    # get the location of the collision
    loc = collision['loc']
    # get the timestep of the collision
    t = collision['timestep']
    # create the two constraints
    if len(loc) == 1:
        # vertex collision
        constraint1 = {'agent': agent, 'loc': loc, 'timestep': t, 'positive': True}
        constraint2 = {'agent': agent, 'loc': loc, 'timestep': t, 'positive': False}
    else:
        # edge collision
        constraint1 = {'agent': a1, 'loc': [loc[1],loc[0]], 'timestep': t, 'positive': False}
        constraint2 = {'agent': a2, 'loc': loc, 'timestep': t, 'positive': False}

    # return the two constraints
    return [constraint1, constraint2]

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        #print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        #print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        #print(root['collisions'])

        # Task 3.2: Testing
        #for collision in root['collisions']:
            #print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # repeat until the open list is empty
        while len(self.open_list) > 0:
            # get the next node from the open list
            node = self.pop_node()
            # check if the node has no collision
            if len(node['collisions']) == 0:
                # return solution
                self.print_results(node)
                return node['paths']
            # otherwise, choose the first collision
            collision = node['collisions'][0]
            # convert to a list of constraints
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)

            # add a new child node to the open list for each constraint
            for constraint in constraints:                
                is_path_valid = True                                  
                # create a copy of the node
                child = {'cost': 0,
                        'constraints': [],
                        'paths': [],
                        'collisions': []}

                # add the constraint to the list of constraints
                child['constraints'] = node['constraints'].copy() + [constraint]   
                # update the paths
                child['paths'] = node['paths'].copy()
                #generate new paths for the agents that violate the constraint
                path = a_star(self.my_map, self.starts[constraint['agent']], self.goals[constraint['agent']], self.heuristics[constraint['agent']], constraint['agent'], child['constraints'])
                if path is not None:
                    # update the paths      
                    child['paths'][constraint['agent']] = path
                    # check if disjoint                 
                    if disjoint:
                        if constraint['positive']:
                            # check if the new path violates any of the constraints
                            agents = paths_violate_constraint(constraint, child['paths'])

                            for agent in agents:
                                child['constraints'].append({'positive': False, 'agent':agent, 'loc':constraint['loc'], 'timestep':constraint['timestep']})                      
                                new_path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, child['constraints'])
                                if new_path is not None:
                                    child['paths'][agent] = new_path
                                else:
                                    is_path_valid = False                                    
                                    break

                    if is_path_valid:
                        # update the cost
                        child['cost'] = get_sum_of_cost(child['paths'])
                        # update the collisions
                        child['collisions'] = detect_collisions(child['paths'])
                        # add the child node to the open list
                        self.push_node(child)

        # if the open list is empty              
        raise BaseException('No solutions')

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
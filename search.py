# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents
from pacman import PacmanRules 

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    from util import Stack
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    # Check if problem == goal state
    initialState = problem.getStartState()
    if problem.isGoalState(initialState):
        return initialState

    # Create LIFO stack
    stack = Stack()
    stack.push((initialState, []))
    visited = []
    
    # Loop
    while(1):
        if(stack.isEmpty()):
            return 0
        node = stack.pop() # is a triple with (state, action, cost)
        state = node[0]
        if state not in visited:
            visited.append(state)
            if problem.isGoalState(state):
                return node[1]

            for neighbor in problem.getSuccessors(state):
                tempPath = node[1].copy()
                neighborState = neighbor[0]
                if neighborState not in visited:
                    tempPath.append(neighbor[1])
                    stack.push((neighborState, tempPath))

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    from util import Queue
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    # Check if problem == goal state
    initialState = problem.getStartState()
    if problem.isGoalState(initialState):
        return initialState

    # Create FIFO queue
    queue = Queue() # create a queue
    queue.push((initialState, [])) # add the node of (initialState, []) to the queue
    visited = [] # keep track of visited nodes
    
    # Loop
    while(1):
        if(queue.isEmpty()): # if queue is empty, return a failure
            return 0
        node = queue.pop() # is a tuple with (state, path)
        state = node[0] # store the state of the current node
        if state not in visited:
            visited.append(state)
            path = node[1] # store the path from start --> current node
            if problem.isGoalState(state): # check if the current node is the goal state
                return path
            for neighbor in problem.getSuccessors(state):
                tempPath = node[1].copy() # keep a temporary path from start --> current node
                neighborState = neighbor[0] # store the neighbor state
                neighborAction = neighbor[1] # store the action from parent --> neighbor

                if neighborState not in visited:
                    tempPath.append(neighborAction) # update the path from start --> neighbor
                    queue.push((neighborState, tempPath)) # push the node (neighborState, tempPath) to the queue

        
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # Check if initial node == goal state
    initialState = problem.getStartState()
    if problem.isGoalState(initialState):
        return initialState


    # Create priority queue
    pq = PriorityQueue()
    pq.push((initialState, []), 0) # push the first node (initial state, []) to the PQ with cost of 0
    visited = [] # keep track of visited nodes
    
    # Loop
    while(1):
        if(pq.isEmpty()):
            return 0
        node = pq.pop() # pop off the node from the PQ
        state = node[0] # store the state of the node
        if state not in visited:
            visited.append(state)
            path = node[1] # store the path from start --> current node
            currPathCost = problem.getCostOfActions(path) # store the cost for the current node
            if problem.isGoalState(state):
                return path # if this node passes the goal test, return the path

            # Loop over all neighbors of the current node
            for neighbor in problem.getSuccessors(state): # neighbor is a triple (successor, action, cost)
                neighborState = neighbor[0] # store the state of the neighbor
                neighborAction = neighbor[1] # store the action to get from parent to neighbor
                neighborCost = neighbor[2] # store the cost to get from parent node to the neighbor node

                tempPath = path.copy() # copy the path to the parent node
                totalCost = currPathCost + neighborCost # cost from start --> neighbor

                if neighborState not in visited:
                    tempPath.append(neighborAction) # new path from start --> neighbor
                    pq.push((neighborState, tempPath), totalCost) # push to PQ the node of (neighborState, tempPath) with cost of pathCost
                


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
     # Check if initial node == goal state
    initialState = problem.getStartState()
    if problem.isGoalState(initialState):
        return initialState


    # Create priority queue
    pq = PriorityQueue()
    pq.push((initialState, []), 0) # push the first node (initial state, []) to the PQ with cost of 0
    visited = [] # keep track of visited nodes
    
    # Loop
    while(1):
        if(pq.isEmpty()):
            return 0
        node = pq.pop() # pop off the node from the PQ
        # print(node)
        state = node[0] # store the state of the node
        if state not in visited:
            path = node[1] # store the path from start --> current node
            visited.append(state)
            currPathCost = problem.getCostOfActions(path) # store the cost for the current node
            if problem.isGoalState(state):
                return path # if this node passes the goal test, return the path

            # Loop over all neighbors of the current node
            for neighbor in problem.getSuccessors(state): # neighbor is a triple (successor, action, cost)
                neighborState = neighbor[0] # store the state of the neighbor
                neighborAction = neighbor[1] # store the action to get from parent to neighbor
                neighborCost = neighbor[2] # store the cost to get from parent node to the neighbor node

                tempPath = path.copy() # copy the path to the parent node
                totalCost = currPathCost + neighborCost + heuristic(neighborState, problem) # cost from start --> neighbor

                if neighborState not in visited:
                    tempPath.append(neighborAction) # new path from start --> neighbor
                    pq.push((neighborState, tempPath), totalCost) # push to PQ the node of (neighborState, tempPath) with cost of pathCost
                
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch



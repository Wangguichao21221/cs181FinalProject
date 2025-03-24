# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        foodlist = newFood.asList()
        if (len(foodlist)==0):
            foodHeuristic = 0
        else:
            mindisToFood = min([manhattanDistance(newPos, food) for food in foodlist])
            foodHeuristic = 10.0/mindisToFood
        mindisToGhost = 999999
        for (ghost,ghostState) in zip(childGameState.getGhostPositions(),newGhostStates):
            if ghostState.scaredTimer <= 0:
                mindisToGhost = min(mindisToGhost,manhattanDistance(newPos, ghost))
        if mindisToGhost<1:
            ghostheuristic = -20
        else:
            ghostheuristic = 0
        "*** YOUR CODE HERE ***"
        return childGameState.getScore()+foodHeuristic+ghostheuristic

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalActions=gameState.getLegalActions(0)

        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            value = self.minimax(gameState.getNextState(0, action), self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction


    def minimax(self, gameState: GameState, depth, agentIndex):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions=gameState.getLegalActions(agentIndex)
        nextStates = [gameState.getNextState(agentIndex, action) for action in legalActions]
        numOfAgents = gameState.getNumAgents()
        if agentIndex == 0:
            return max([self.minimax(nextState, depth, agentIndex+1) for nextState in nextStates])
        else:
            if agentIndex == numOfAgents-1:
                return min([self.minimax(nextState, depth-1, 0) for nextState in nextStates])
            else:
                return min([self.minimax(nextState, depth, agentIndex+1) for nextState in nextStates])
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        with alpha-beta pruning.
        """
        
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')

        for action in legalActions:
            value = self.alphaBeta(gameState.getNextState(0, action), self.depth, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
    def alphaBeta(self,gameState:GameState, depth, agentIndex, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalActions = gameState.getLegalActions(agentIndex)
            if agentIndex == 0: 
                value = float('-inf')
                for action in legalActions:
                    value = max(value, self.alphaBeta(gameState.getNextState(agentIndex, action), depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else: 
                value = float('inf')
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth -= 1
                for action in legalActions:
                    value = min(value, self.alphaBeta(gameState.getNextState(agentIndex, action), depth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions=gameState.getLegalActions(0)

        bestAction = None
        bestValue = float('-inf')
        if len(legalActions)==0:
            util.raiseNotDefined()
        for action in legalActions:
            value = self.expectimax(gameState.getNextState(0, action), self.depth, 1)
            # print(value)
            if value > bestValue:
                bestValue = value
                bestAction = action
        if bestAction==None:
            print(legalActions)
            print(bestValue)
            print(self.depth)
            util.raiseNotDefined()
        return bestAction

    def expectimax(self, gameState: GameState, depth, agentIndex):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions=gameState.getLegalActions(agentIndex)
        nextStates = [gameState.getNextState(agentIndex, action) for action in legalActions]
        numOfAgents = gameState.getNumAgents()
        if agentIndex == 0:
            return max([self.expectimax(nextState, depth, agentIndex+1) for nextState in nextStates])
        else:
            if agentIndex == numOfAgents-1:
                return sum([self.expectimax(nextState, depth-1, 0) for nextState in nextStates]) / len(legalActions)
            else:
                return sum([self.expectimax(nextState, depth, agentIndex+1) for nextState in nextStates]) / len(legalActions)

    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()
    if currentGameState.isWin():
        if (len(capsules)==0):
            return currentGameState.getScore()
        else:
            return currentGameState.getScore() - 200
    if currentGameState.isLose():
        return currentGameState.getScore()
    currentScore = currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodHeuristic = 0
    if foodList:
        minDistanceToFood = min([bfs(currentGameState,pacmanPosition, food) for food in foodList])
        foodHeuristic = 9.0 / minDistanceToFood
    mindisToGhost = 999999
    ghostheuristic = 0
    for ghost in currentGameState.getGhostPositions():
        if currentGameState.getGhostState(1).scaredTimer <= 0:
            mindisToGhost = min(mindisToGhost,manhattanDistance(currentGameState.getPacmanPosition(), ghost))
        else:
            ghostheuristic += 200 / manhattanDistance(currentGameState.getPacmanPosition(), ghost)
    if mindisToGhost< 1:
        ghostheuristic -=20
    capsuleHeuristic = 0
    if capsules:
        minDistanceToCapsule = min([bfs( currentGameState,pacmanPosition, capsule) for capsule in capsules])
        capsuleHeuristic = 18.0 / minDistanceToCapsule

    return currentScore + foodHeuristic + ghostheuristic + capsuleHeuristic
from util import Queue
def bfs(gameState, start, goal):
    """
    Perform BFS to find the shortest path from start to goal.
    """
    if start == goal:
        return 0

    walls = gameState.getWalls()
    visited = set()
    queue = Queue()
    queue.push((start, 0))

    while not queue.isEmpty():
        (currentPosition, currentDistance) = queue.pop()
        if currentPosition in visited:
            continue
        visited.add(currentPosition)

        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(direction)
            nextPosition = (int(x + dx), int(y + dy))

            if nextPosition == goal:
                return currentDistance + 1

            if not walls[nextPosition[0]][nextPosition[1]]:
                queue.push((nextPosition, currentDistance + 1))

    return float('inf')  # If no path is found
# Abbreviation
better = betterEvaluationFunction
def fastbetterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()
    if currentGameState.isWin():
        if (len(capsules)==0):
            return currentGameState.getScore()
        else:
            return currentGameState.getScore() - 20
    if currentGameState.isLose():
        return currentGameState.getScore() - 500
    currentScore = currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    foodHeuristic = 0
    if foodList:
        minDistanceToFood = min([manhattanDistance(pacmanPosition, food) for food in foodList])
        foodHeuristic = 15.0 / minDistanceToFood
    mindisToGhost = 999999
    ghostheuristic = 0
    ghosts = currentGameState.getGhostStates()
    for ghost in ghosts:
        if ghost.scaredTimer <= 0:
            mindisToGhost = min(mindisToGhost,manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()))
    if mindisToGhost< 3:
        ghostheuristic -=20
    elif mindisToGhost< 2:
        ghostheuristic -=40
    elif mindisToGhost< 1:
        ghostheuristic -=80
    capsuleHeuristic = 0
    numOfscaredGhosts = 0
    for ghost in ghosts:
        if ghost.scaredTimer > 0:
            numOfscaredGhosts += 1
    if capsules and numOfscaredGhosts == 0:
        minDistanceToCapsule = min([manhattanDistance(pacmanPosition, capsule) for capsule in capsules])
        capsuleHeuristic = 85.0 / minDistanceToCapsule
    return currentScore + foodHeuristic + ghostheuristic + capsuleHeuristic
class ContestAgent(ExpectimaxAgent):
    """
      Your agent for the mini-contest
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = fastbetterEvaluationFunction
        self.depth = int(depth)
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions=gameState.getLegalActions(0)

        bestAction = None
        bestValue = float('-inf')
        if len(legalActions)==0:
            util.raiseNotDefined()
        for action in legalActions:
            value = self.expectimax(gameState.getNextState(0, action), self.depth, 1)
            # print(value)
            if value > bestValue:
                bestValue = value
                bestAction = action
        if bestAction==None:
            print(legalActions)
            print(bestValue)
            print(self.depth)
            util.raiseNotDefined()
        return bestAction

    def expectimax(self, gameState: GameState, depth, agentIndex):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions=gameState.getLegalActions(agentIndex)
        nextStates = [gameState.getNextState(agentIndex, action) for action in legalActions]
        numOfAgents = gameState.getNumAgents()
        if agentIndex == 0:
            return max([self.expectimax(nextState, depth, agentIndex+1) for nextState in nextStates])
        else:
            if agentIndex == numOfAgents-1:
                return sum([self.expectimax(nextState, depth-1, 0) for nextState in nextStates]) / len(legalActions)
            else:
                return sum([self.expectimax(nextState, depth, agentIndex+1) for nextState in nextStates]) / len(legalActions)

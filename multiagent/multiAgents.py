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
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #___main__.GameState
        newPos = successorGameState.getPacmanPosition() # tuple
        newFood = successorGameState.getFood() # game.Grid
        newGhostStates = successorGameState.getGhostStates() # list
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] # list

        "*** YOUR CODE HERE ***"
        score = currentGameState.getScore()

        newFoodList = newFood.asList()
        minDistanceToFood = 0
        for food in newFoodList:
            foodDistance = manhattanDistance(food, newPos)
            if foodDistance > minDistanceToFood:
                minDistanceToFood = foodDistance
        score += 10 / (minDistanceToFood + 0.1)
        
        ghostPositions = successorGameState.getGhostPositions()
        minDistanceToGhost = 0
        for ghost in ghostPositions:
            ghostDistance = manhattanDistance(ghost, newPos)
            if ghostDistance > minDistanceToGhost:
                minDistanceToGhost = ghostDistance
        score -= 8 / (minDistanceToGhost + 0.1)

        capsulesList = successorGameState.getCapsules()
        minDistanceToCapsule = 0
        for capsule in capsulesList:
            capsuleDistance = manhattanDistance(capsule, newPos)
            if capsuleDistance > minDistanceToCapsule:
                minDistanceToCapsule = capsuleDistance
        score += 5 / (minDistanceToCapsule + 0.1)

        score -= 4 * len(newFoodList)

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxVal(gameState, 0, 0)[1]

    def minimax(self, gameState, agentIndex, depth):
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, depth)
        else:
            return self.minVal(gameState, agentIndex, depth)
        
    def maxVal(self, currGameState, agentIndex, depth):
        bestScore = float('-inf')
        bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(succGameState, 1, depth)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

    def minVal(self, currGameState, agentIndex, depth):
        bestScore = float('inf')
        bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)

            if agentIndex == currGameState.getNumAgents() - 1:
                score, _ = self.minimax(succGameState, 0, depth + 1)
            else:
                score, _ = self.minimax(succGameState, agentIndex + 1, depth)
            if score < bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxVal(gameState, 0, 0, float('-inf'), float('inf'))[1]

    def alphaBeta(self, gameState, agentIndex, depth, a, b):
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, depth, a, b)
        else:
            return self.minVal(gameState, agentIndex, depth, a, b)
        
    def maxVal(self, currGameState, agentIndex, depth, a, b):
        bestScore = float('-inf')
        bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)
            score, _ = self.alphaBeta(succGameState, 1, depth, a, b)
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            if bestScore > b:
                return bestScore, bestAction
            a = max(a, bestScore)

        return bestScore, bestAction

    def minVal(self, currGameState, agentIndex, depth, a, b):
        bestScore = float('inf')
        bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)

            if agentIndex == currGameState.getNumAgents() - 1:
                score, _ = self.alphaBeta(succGameState, 0, depth + 1, a, b)
            else:
                score, _ = self.alphaBeta(succGameState, agentIndex + 1, depth, a, b)
            if score < bestScore:
                bestScore = score
                bestAction = action

            if bestScore < a:
                return bestScore, bestAction
            b = min(b, bestScore)

        return bestScore, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxVal(gameState, 0, 0)[1]

    def expectimax(self, gameState, agentIndex, depth):
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, depth)
        else:
            return self.expVal(gameState, agentIndex, depth)
        
    def maxVal(self, currGameState, agentIndex, depth):
        bestScore = float('-inf')
        bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)
            score, _ = self.expectimax(succGameState, 1, depth)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction

    def expVal(self, currGameState, agentIndex, depth):
        totalScore = 0
        # bestAction = None

        for action in currGameState.getLegalActions(agentIndex):
            succGameState = currGameState.generateSuccessor(agentIndex, action)

            if agentIndex == currGameState.getNumAgents() - 1:
                score, _ = self.expectimax(succGameState, 0, depth + 1)
            else:
                score, _ = self.expectimax(succGameState, agentIndex + 1, depth)
            totalScore += score
        
        avgScore = totalScore / len(currGameState.getLegalActions(agentIndex))

        return avgScore, None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

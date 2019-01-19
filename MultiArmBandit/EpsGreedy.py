import numpy as np
from MAB import MAB

class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    local_narms = 0
    local_epsilon = 0.0
    local_Q0 = np.inf
    round_reward_estimatedvalue = []

    def __init__(self, narms, epsilon, Q0=np.inf):

        # check if narms is positive else throw error
        assert narms > 1, "narms should be atleast 1 or more"
        self.local_narms = narms

        # check if eplison is between 0 and 1
        assert 0 < epsilon < 1, "epsilon should be between 0 and 1"
        self.local_epsilon = epsilon

        self.local_Q0 = Q0

        # create a matrix to store Round, award and Estimate Value of awards(Q)
        self.round_reward_estimatedvalue = []

        for a in range(1, self.local_narms + 1):
            self.round_reward_estimatedvalue.append([0, 0, 0.0])

    def play(self, tround, context=None):

        Q = 0.0
        best_arms = []

        # Choose whether to eploit or explore

        if (np.random.uniform() > self.local_epsilon):  # lets exploit

            # choose the best available known arm from the historical data (round_reward_estimatedvalue),
            # slice the estimatedValue, and find max value

            Q = np.amax(np.array(self.round_reward_estimatedvalue)[:, [2]])

            # Check if it ties with any other arms, if it does, make a list of tied arms

            for i in range(len(self.round_reward_estimatedvalue)):
                if (Q == self.round_reward_estimatedvalue[i][2]):
                    best_arms.append(i)

            # if it is a tie, choose randomly any one of the arm
            # randint chooses uniform discreet integers

            return (best_arms[np.random.randint(best_arms.__len__())] + 1)


        else:  # lets explore choose a random arm

            return (np.random.randint(self.local_narms) + 1)

    def update(self, arm, reward, context=None):

        # take care of indexing(starts from 0)
        arm = arm - 1

        # update the tround (increment the round)
        self.round_reward_estimatedvalue[arm][0] = self.round_reward_estimatedvalue[arm][0] + 1

        # increment the # of time played and store
        self.round_reward_estimatedvalue[arm][1] = self.round_reward_estimatedvalue[arm][1] + reward

        # Update the Q value
        self.round_reward_estimatedvalue[arm][2] = self.round_reward_estimatedvalue[arm][1] / float(
            self.round_reward_estimatedvalue[arm][0])

# Do not edit. These are the only imports permitted.
import numpy as np
from MAB import MAB

class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """

    local_narms = 0
    local_rho = 0.0
    local_Q0 = np.inf
    round_reward_estimatedvalue = []

    def __init__(self, narms, rho, Q0=np.inf ):

        self.local_Q0 = np.inf

        # check if narms is positive else throw error
        assert narms > 1, "narms should be atleast 1 or more"
        self.local_narms = narms

        self.local_rho = rho

        self.round_reward_estimatedvalue = []

        # create a matrix to store Round, award and Estimate Value of awards(Q)
        for a in range(1, self.local_narms + 1):
            self.round_reward_estimatedvalue.append([0, 0, 0.0])

    def play(self, tround, context=None):

        Q = 0.0
        best_arms = []

        ''' For each arm calculate the Q based on tround
           1. get Nt 
           2. Calculate mean
           3. mean + squareroot of (2*log(tround)/Nt)
        '''

        for i in range(len(self.round_reward_estimatedvalue)):

            # 1 . get Nt
            Nt = self.round_reward_estimatedvalue[i][0]

            if (Nt == 0):
                self.round_reward_estimatedvalue[i][2] = self.local_Q0

            else:
                # 2 . Calculate Mean
                mean = (self.round_reward_estimatedvalue[i][1]) / float(Nt)

                # 3 . calculate Q and store it in round_reward_estimatedvalue
                self.round_reward_estimatedvalue[i][2] = mean + np.sqrt((self.local_rho * np.log(tround)) / float(Nt))

        # Select the arm with max UCB (average reward + Q(n))
        Q = np.amax(np.array(self.round_reward_estimatedvalue)[:, [2]])

        # Check if it ties with any other arms, if it does, make a list of tied arms
        for i in range(len(self.round_reward_estimatedvalue)):
            if (Q == self.round_reward_estimatedvalue[i][2]):
                best_arms.append(i)

        # if it is a tie, choose randonly any one of the arm
        # randint chooses uniform discreet integers

        return (best_arms[np.random.randint(best_arms.__len__())] + 1)

    def update(self, arm, reward, context=None):

        # take care of indexing(starts from 0)
        arm = arm - 1

        # update the tround (increment the round)
        self.round_reward_estimatedvalue[arm][0] = self.round_reward_estimatedvalue[arm][0] + 1

        # update the reward
        self.round_reward_estimatedvalue[arm][1] = self.round_reward_estimatedvalue[arm][1] + reward
        # print (self.round_reward_estimatedvalue)

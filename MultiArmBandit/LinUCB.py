import numpy as np
from MAB import MAB

class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    local_narms = 0
    local_ndims = 0
    local_alpha = 1.0
    local_context_2D = None
    A_b_p = []

    def __init__(self, narms, ndims, alpha):

        assert narms > 1, "Number of arms should always be more than 1"
        self.local_narms = narms

        assert narms > 1, "Number of arms should always be more than 1"
        self.local_ndims = ndims

        self.local_alpha = alpha

        '''create a matrix to store 
            1. A - m*d dimensional matrix
            2. b - vector to store actual returns for each arm
            3. p - Estimated payoff value 

        '''
        self.A_b_p = []
        for a in range(1, self.local_narms + 1):
            self.A_b_p.append([np.identity(self.local_narms), np.zeros(self.local_narms), np.inf])

    def play(self, tround, context):

        best_arms = []
        theta = None
        arms = 0
        self.local_context_2D = np.reshape(context.astype(np.float), (10, 10))

        for each in self.A_b_p:  # apply context for each arm

            theta = np.matmul(np.linalg.inv(self.A_b_p[arms][0]), self.A_b_p[arms][1])

            x = np.matmul(self.local_context_2D[arms].transpose(), np.linalg.inv(self.A_b_p[arms][0]))

            self.A_b_p[0][2] = np.matmul(theta.transpose(), self.local_context_2D[arms]) + (
                        self.local_alpha * np.sqrt(np.matmul(x, self.local_context_2D[arms])))

            arms = arms + 1

            # choose the max of A_b_p and return the best arm.

        maxValue = np.amax(np.array(self.A_b_p)[:, [2]])

        # Check if it ties with any other arms, if it does, make a list of tied arms
        for i in range(len(self.A_b_p)):
            if (maxValue == self.A_b_p[i][2]):
                best_arms.append(i)

        # if it is a tie, choose randonly any one of the arm
        # randint chooses uniform discreet integers

        return (best_arms[np.random.randint(best_arms.__len__())] + 1)

    def update(self, arm, reward, context):

        c = np.reshape(context.astype(np.float), (10, 10))

        # indexing
        arm = arm - 1

        # Update A

        self.A_b_p[arm][0] = np.add(self.A_b_p[arm][0],
                                    np.matmul(c[arm].astype(np.float), c[arm].astype(np.float).transpose()))

        # update b
        self.A_b_p[arm][1] = np.add(self.A_b_p[arm][1], (reward * np.array(c[arm].astype(np.float))))

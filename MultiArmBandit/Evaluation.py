import numpy as np
import matplotlib.pyplot as plt
from UCB import UCB
from LinUCB import LinUCB
from EpsGreedy import EpsGreedy


def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """
    history_h = []
    payoff_R = []
    T = 1
    event = 0  # event = each record in the file

    while ((T < nrounds) & (event < arms.__len__())):  # keep running the loop until either
        # nrounds are completed or the events are run out(10,000)

        if (int(mab.play(T, contexts[event])) == int(
                arms[event])):  # if the played arm is same as the event, match found!

            mab.update(int(arms[event]), int(rewards[event]), contexts[event])  # update the mab
            history_h.append(int(arms[event]))
            payoff_R.append(int(rewards[event]))  # keep a track of payoff

            T = T + 1  # since the match is found, update the round

        event = event + 1

    return payoff_R


'''
This piece of code reads the file "dataset.txt" and prepares numpy arrays of the arms, rewards and context.
These would be used by the following method calls.

please ensure the dataset file is placed in the same location as this notebook.
'''

temp_arms = []
temp_rewards = []
temp_contexts = []
i = 0
for line in open("dataset.txt"):  # read from the file

    listWords = line.rstrip().split(" ")  # get rid of "\n" and additional spaces
    temp_arms.append(listWords[0])  # strip out first column -- arms
    temp_rewards.append(listWords[1])  # strip out second column -- rewards
    temp_contexts.append(listWords[2:])  # strip out 3  - 102 columns -- contexts

arms = np.array(temp_arms)  # convert into numpy array
rewards = np.array(temp_rewards)  # convert into numpy array
contexts = np.reshape(np.array(temp_contexts),
                      (10000, 100))  # form an numpy array of [10000]*[10*10] [events]*[arms*features]



mab = EpsGreedy(10, 0.05)
results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('EpsGreedy average reward', np.mean(results_EpsGreedy))


mab = UCB(10, 1.0)
results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('UCB average reward', np.mean(results_UCB))


mab = LinUCB(10, 10, 1.0)
results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('LinUCB average reward', np.mean(results_LinUCB))


plt.plot(np.cumsum(results_EpsGreedy) , label = "Epsilon-Greedy")
plt.plot(np.cumsum(results_UCB), label = "UCB")
plt.plot(np.cumsum(results_LinUCB), label = "LinUCB")
plt.ylabel('Cumulative sum(r(t,a))')
plt.xlabel('Rounds(T)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
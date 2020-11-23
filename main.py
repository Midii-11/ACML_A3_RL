import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_discrete_state(env, state, discrete_observation_window):
    discrete_state = (state - env.observation_space.low) / discrete_observation_window
    return tuple(discrete_state.astype(np.int))


def rendering(episode, showEvery):
    if episode % showEvery == 0:
        print(episode)
        render = True
    else:
        render = False
    return render


def heatmap(p_domain, v_domain, Q, i):
    H = np.zeros([p_domain, v_domain])
    A = np.zeros([p_domain, v_domain])
    for p in range(p_domain):
        for v in range(v_domain):
            m = Q[p][v][0]
            a = 0
            if Q[p][v][1] > m:
                m = Q[p][v][1]
                a = 1
            if Q[p][v][1] > m:
                m = Q[p][v][1]
                a = 2
            H[p][v] = m
            A[p][v] = a

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    cmap = sns.color_palette("RdBu_r", 7)
    sns.heatmap(H, cmap=cmap)
    plt.title('Heatmap of Q*')
    plt.xlabel('Velocity')
    plt.ylabel('Position')

    plt.subplot(1, 2, 2)
    cmap = sns.color_palette("RdBu_r", 7)
    sns.heatmap(A, cmap=cmap)
    plt.title('Action map')
    plt.xlabel('Velocity')
    plt.ylabel('Position')
    print(epsilon[i])
    path = ('Figures\\Fig_' + str(epsilon[i]) + ".png")
    plt.savefig(path)
    # plt.show()


def run(q_table_size, episodes, eps, i):
    env = gym.make("MountainCar-v0")

    episodes = episodes
    showEvery = episodes // 10
    learningRate = 0.1
    # Remember: determines how important the current reward is compared to future rewards
    discount = 0.95
    # Remember: determine how often we explore
    epsilon = eps
    start_epsilon_decay = 1
    end_epsilon_decay = episodes // 2
    epsilon_decay_val = epsilon / (end_epsilon_decay - start_epsilon_decay)

    # Remember: make env discrete (env.obs_space_delta / discrete_env_size)
    #           --> env made of 20 observation points of size: [0.09  0.007]
    discrete_observation_size = [q_table_size] * len(env.observation_space.high)
    discrete_observation_window = (env.observation_space.high - env.observation_space.low) / discrete_observation_size

    # Remember: Initialize table with negative values (since reward is -1 or 0)
    #           shape = (20, 20, 3)
    #           size = 1200 combinations of state-reward
    q_table = np.random.uniform(low=-2, high=0, size=(discrete_observation_size + [env.action_space.n]))

    for episode in range(episodes):
        # render every X episodes
        render = rendering(episode, showEvery)

        # get initial discrete state
        discrete_state = get_discrete_state(env, env.reset(), discrete_observation_window)

        # explore environment and update Q_table
        done = False
        while not done:

            # explore new move or apply known move based on epsilon
            if np.random.random() > epsilon:
                # Remember: 0 = left | 1 = none | 2 = right
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Remember:
            #   new_state[pos, vel]
            #   reward = -1 (except flag = 0)
            #   done = False (except conditions)
            # get new state (after a given action)
            new_state, reward, done, _ = env.step(action)
            # Remember: new_state = continuous :/ --> need discrete
            # get new discrete state
            new_discrete_state = get_discrete_state(env, new_state, discrete_observation_window)

            # render every X episodes
            if render:
                env.render()

            # if goal is not reached update Q_table with new q value
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]

                # Remember: apply Q-learning formula (see picture)
                new_q = (1 - learningRate) * current_q + learningRate * (reward + discount * max_future_q)
                # Remember: update q_table
                q_table[discrete_state + (action,)] = new_q
            # if goal is reached update Q_table with 0
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0

            # update value after choosing a move
            discrete_state = new_discrete_state

        # update epsilon after each episode
        if end_epsilon_decay >= episode >= start_epsilon_decay:
            epsilon -= epsilon_decay_val

    heatmap(p_domain=discrete_observation_size[0], v_domain=discrete_observation_size[1], Q=q_table, i=i)
    env.close()


if __name__ == '__main__':
    np.random.seed(42)
    q_table_size = 80
    episodes = 25000
    epsilon = [0.01, 0.1, 0.5, 0.9, 9, 30]

    for i in range(len(epsilon)):
        run(q_table_size, episodes, epsilon[i], i)

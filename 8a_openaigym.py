#### Reinforcement learning basic scripts

case_switch=2

if (case_switch==1):
    import gym
    env = gym.make("CartPole-v1")
    observation = env.reset()
    for _ in range(200):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
elif(case_switch==2):
    import numpy as np
    nan = np.nan
    actions = [[0, 1, 2], [0, 2], [0]]
    P = np.array([
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]],
        [[0.8, 0.2, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [nan, nan, nan], [nan, nan, nan]],
        ])
    R = np.array([
        [[20., 0.0, 0.0], [0.0, 0.0, 0.0], [-10., -10., 0.0]],
        [[40., 30., 0.0], [nan, nan, nan], [0.0, 0.0, -10.]],
        [[70., 0.0, 0.0], [nan, nan, nan], [nan, nan, nan]],
        ])
    Q = np.full((3, 3), -np.inf)
    for s, a in enumerate(actions):
        Q[s, a] = 0.0
 
    discount_factor = 0.999
    iterations = 10
    for i in range(iterations):
        Q_previous = Q.copy()
        for s in range(len(P)):
            for a in actions[s]:
                sum_v = 0
                for s_next in range(len(P)):
                    sum_v += P[s, a, s_next] * (R[s, a, s_next] + discount_factor * np.max(Q_previous[s_next]))
                Q[s, a] = sum_v
    print('row:states',' action:columns')
    print(Q)
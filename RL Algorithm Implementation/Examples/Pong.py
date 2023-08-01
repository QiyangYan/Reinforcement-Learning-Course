import gymnasium as gym
env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset()

print(env.get_action_meanings())

scores = 0

for _ in range(3000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        print("Episode finished after {} steps".format(scores))
        observation, info = env.reset()
        scores = 0
    else:
        scores +=1

env.close()
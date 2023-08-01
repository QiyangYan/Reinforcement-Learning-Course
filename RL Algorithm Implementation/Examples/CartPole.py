import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

print(env.action_space)

observation, info = env.reset(seed=42)
steps = 0
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        print("Episode finished after {} steps".format(steps))
        observation, info = env.reset()
        steps = 0
    else:
        steps += 1

env.close()
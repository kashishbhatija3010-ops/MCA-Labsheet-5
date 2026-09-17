# MCA-Labsheet-5
Lab Assessment on Reinforcement Learning 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

# pip install gymnasium
# Q1
print("Gymnasium version:", gym.__version__)
print("Gymnasium is installed and configured successfully.")

# Q2
env = gym.make("CartPole-v1")
observation, info = env.reset()
print("Initial Observation:")
print(observation)

# Q3
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Number of Actions:", env.action_space.n)

# Q4
observation, info = env.reset()
print("Initial State:", observation)
for step in range(5):
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
    print("\nStep:", step + 1)
    print("Action:", action)
    print("Reward:", reward)
    print("Next State:", next_observation)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    if terminated or truncated:
        break
    observation = next_observation
    
env.close()

# Q5
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
print("Initial State:", state)
total_reward = 0
for step in range(20):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step: {step + 1}, "
        f"Action: {action}, "
        f"State: {next_state}, "
        f"Reward: {reward}"
    )
    total_reward += reward
    state = next_state
    if terminated or truncated:
        print("\nEpisode finished.")
        break
print("Total Reward:", total_reward)

# Q6
q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 10000
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Q-Learning completed.")

# Q7
q_table_df = pd.DataFrame(
    q_table,
    columns=["Left", "Down", "Right", "Up"]
)
print(q_table_df)

# Q8
success = 0
test_episodes = 1000
for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if reward == 1:
            success += 1
print("Success Rate:", (success / test_episodes) * 100, "%")

# Q9
actions = ["Left", "Down", "Right", "Up"]
for state in range(env.observation_space.n):
    best_action = actions[np.argmax(q_table[state])]
    print("State", state, "->", best_action)

# Q10
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    state = next_state
    done = terminated or truncated
print("Total Reward:", total_reward)
print("Final State:", state)

# Q11
policy = np.argmax(q_table, axis=1)
for state, action in enumerate(policy):
    print("State", state, "->", actions[action])

# Q12
for episode in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    print("Episode", episode + 1, "Reward:", total_reward)

# Q13
rewards = []
for episode in range(100):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    rewards.append(total_reward)
print("Average Reward:", np.mean(rewards))

# Q14
successful_episodes = sum(rewards)
print("Successful Episodes:", int(successful_episodes))
print("Total Episodes:", len(rewards))

# Q15
policy_grid = np.array([actions[action][0] for action in policy]).reshape(4, 4)
print(policy_grid)

# Q16
random_success = 0
q_success = 0
for episode in range(1000):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    random_success += reward
for episode in range(1000):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    q_success += reward
print("Random Agent Success Rate:", random_success / 10, "%")
print("Q-Learning Success Rate:", q_success / 10, "%")

# Q17
state, _ = env.reset()
path = [state]
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    path.append(state)
    done = terminated or truncated
print("Path:", path)
print("Reward:", reward)

# Q18
print("Number of Steps:", len(path) - 1)

# Q19
learning_rates = [0.1, 0.5, 0.8, 1.0]
results = {}

for lr in learning_rates:
    q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(5000):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() if np.random.rand() < 0.1 else np.argmax(q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            q[state, action] += lr * (
                reward + gamma * np.max(q[next_state]) - q[state, action]
            )
            state = next_state
    results[lr] = q
for lr, q in results.items():
    print("Learning Rate:", lr, "Maximum Q-value:", np.max(q))

# Q20
final_q_table = pd.DataFrame(
    q_table,
    columns=["Left", "Down", "Right", "Up"]
)
print(final_q_table.round(3))

# Q21
plt.figure(figsize=(10, 5))
plt.plot(q_table)
plt.xlabel("State")
plt.ylabel("Q-value")
plt.title("Q-values for All States")
plt.legend(actions)
plt.show()

# Q22
max_q_values = np.max(q_table, axis=1)
for state, value in enumerate(max_q_values):
    print("State", state, "->", round(value, 3))

# Q23
for state in range(len(q_table)):
    best_action = np.argmax(q_table[state])
    print("State:", state, "Action:", actions[best_action],
          "Q-value:", round(q_table[state, best_action], 3))

# Q24
success = 0
episodes = 1000

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    success += reward
print("Success Rate:", success / episodes * 100, "%")

# Q25
steps = []

for episode in range(100):
    state, _ = env.reset()
    done = False
    count = 0
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        count += 1
    steps.append(count)
print("Average Steps:", np.mean(steps))

# Q26
policy_grid = np.array([
    actions[np.argmax(q_table[state])][0]
    for state in range(16)
]).reshape(4, 4)

print(policy_grid)

# Q27
success = 0
for episode in range(20):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    success += reward
    print("Episode", episode + 1, "Reward:", reward)
print("Successful Episodes:", int(success))

# Q28
rewards = []

for episode in range(100):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    rewards.append(total_reward)
print("Average Reward:", np.mean(rewards))

# Q29
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards During Testing")
plt.show()

# Q30
success_rate = np.mean(rewards) * 100
average_steps = np.mean(steps)
print("Final Performance")
print("Success Rate:", success_rate, "%")
print("Average Steps:", average_steps)

# Q31
plt.figure(figsize=(8, 6))
plt.imshow(q_table)
plt.colorbar()
plt.xlabel("Actions")
plt.ylabel("States")
plt.xticks(range(4), actions)
plt.title("Q-Table Heatmap")
plt.show()

# Q32
state = np.unravel_index(np.argmax(q_table), q_table.shape)
print("State:", state[0])
print("Action:", actions[state[1]])
print("Highest Q-value:", q_table[state])

# Q33
start_state = 0
best_action = np.argmax(q_table[start_state])
print("Starting State:", start_state)
print("Best Action:", actions[best_action])
print("Q-value:", q_table[start_state, best_action])

# Q34
state, _ = env.reset()
done = False
path = [state]
while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    path.append(state)
    done = terminated or truncated
print("Optimal Path:", path)
print("Final Reward:", reward)

# Q35
print("Q-Learning Agent Results")
print("------------------------")
print("Success Rate:", success_rate, "%")
print("Average Steps:", average_steps)
print("Optimal Path:", path)
print("Final Reward:", reward)

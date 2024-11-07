from grid_world_env import GridWorldEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    env = GridWorldEnv(grid_width=15,
                       grid_height=10,
                       player_start_pos=[2, 5],
                       render_mode=None)

    # Wrap the environment with a Monitor wrapper to record episode statistics
    env = Monitor(env)
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)

    total_timesteps = 10000
    print(f"Training the model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed.")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # ---------------------
    print("Running inference with the trained agent...")
    env = GridWorldEnv(grid_width=15,
                       grid_height=10,
                       player_start_pos=[2, 5],
                       render_mode='human')

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Use the trained model to predict the next action
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)  # Ensure action is an integer
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        # The environment will render automatically because render_mode='human'

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    env.close()

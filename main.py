from utils import load_config, get_env_properties
from dreamer import Dreamer

from dm_control import suite
from dm_control.suite.wrappers import pixels


def main(config_file):
    env_train    = suite.load(domain_name="cartpole", task_name="swingup")
    env_train    = pixels.Wrapper(env=env_train, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    env_eval     = suite.load(domain_name="cartpole", task_name="swingup")
    env_eval     = pixels.Wrapper(env=env_eval, render_kwargs={"height": 64, "width": 64, "camera_id": 0})

    observation_shape, action_size, action_min, action_max = get_env_properties(env=env_train)

    config = load_config("CartPoleSwingUp_Dreamer_v1")

    print(observation_shape, action_size, action_min, action_max)
    print(observation_shape)
    print(*observation_shape)

    dreamer = Dreamer(observation_shape, action_size, config.dreamer)

    dreamer.environement_interaction(env_train, config.nb_episodes_before_start, config.seed)

    iterations_nb = config.gradient_steps // config.replay_ratio 
    for _ in range(iterations_nb):
        for _ in range(config.replay_ratio):
            sampled_data = dreamer.buffer.sample(dreamer.config.batch_size, dreamer.config.batch_length)
            


if __name__ == "__main__":
    main()
    

    

    
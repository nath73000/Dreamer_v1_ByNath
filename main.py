from utils import load_config, get_env_properties
from dreamer import Dreamer

from dm_control import suite
from dm_control.suite.wrappers import pixels






if __name__ == "__main__":

    env_train    = suite.load(domain_name="cartpole", task_name="swingup")
    env_train    = pixels.Wrapper(env=env_train, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    env_test     = suite.load(domain_name="cartpole", task_name="swingup")
    env_test     = pixels.Wrapper(env=env_test, render_kwargs={"height": 64, "width": 64, "camera_id": 0})

    observation_shape, action_size, action_min, action_max = get_env_properties(env=env_train)

    config = load_config("CartPoleSwingUp_Dreamer_v1")

    print(observation_shape, action_size, action_min, action_max)

    #dreamer = Dreamer()
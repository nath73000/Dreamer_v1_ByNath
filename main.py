from utils import load_config, get_env_properties, plotMetrics, saveLossesToCSV
from dreamer import Dreamer

from dm_control import suite
from dm_control.suite.wrappers import pixels
import os


def main(config_file):
    env_train    = suite.load(domain_name="cartpole", task_name="swingup")
    env_train    = pixels.Wrapper(env=env_train, render_kwargs={"height": 64, "width": 64, "camera_id": 0})
    env_eval     = suite.load(domain_name="cartpole", task_name="swingup")
    env_eval     = pixels.Wrapper(env=env_eval, render_kwargs={"height": 64, "width": 64, "camera_id": 0})

    observation_shape, action_size, action_min, action_max = get_env_properties(env=env_train)

    config = load_config("CartPoleSwingUp_Dreamer_v1")
    runName                   = f"{config.environment_name}_{config.run_name}"
    checkpoint_to_load        = os.path.join(config.folder_names.checkpoints_folder, f"{runName}_{config.checkpoint_to_load}")
    metrics_file_name         = os.path.join(config.folder_names.metrics_folder,        runName)
    plot_file_name            = os.path.join(config.folder_names.plots_folder,          runName)
    checkpoint_filename_base  = os.path.join(config.folder_names.checkpoints_folder,    runName)
    video_filename_base       = os.path.join(config.folder_names.videos_folder,         runName)

    dreamer = Dreamer(observation_shape, action_size, config.dreamer)

    dreamer.environement_interaction(env_train, config.nb_episodes_before_start, config.seed)

    iterations_nb = config.gradient_steps // config.replay_ratio 
    for _ in range(iterations_nb):
        for _ in range(config.replay_ratio):
            sampled_data                        = dreamer.buffer.sample(dreamer.config.batch_size, dreamer.config.batch_length)
            initial_states, world_model_metrics = dreamer.world_model_training(sampled_data)
            behavior_metrics                    = dreamer.behavior_training(initial_states)
            dreamer.total_gradient_steps += 1

            if dreamer.total_gradient_steps % config.checkpoint_interval == 0 and config.save_checkpoints:
                suffix = f"{dreamer.total_gradient_steps/1000:.0f}k"
                dreamer.save_checkpoint(f"{checkpoint_filename_base}_{suffix}")
                evaluation_score = dreamer.environment_interaction(env_eval, config.num_evaluation_episodes, seed=config.seed, evaluation=True, save_video=True, filename=f"{video_filename_base}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluation_score:>8.2f}")

        most_recent_score = dreamer.environment_interaction(env_train, config.num_interaction_episodes, seed=config.seed)
        if config.save_metrics:
            metrics_base = {"env_steps": dreamer.total_env_steps, "gradient_steps": dreamer.total_gradient_steps, "total_reward" : most_recent_score}
            saveLossesToCSV(metrics_file_name, metrics_base | world_model_metrics | behavior_metrics)
            plotMetrics(f"{metrics_file_name}", savePath=f"{plot_file_name}", title=f"{config.environment_name}")


if __name__ == "__main__":
    
    main()
    

    

    
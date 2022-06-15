import os
import shutil
import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward




class YoutubeExampleBot(object):
    def __init__(self,
                 frame_skip = 8,
                 half_life_seconds=5,
                 fps=120,
                 agents_per_match=6,
                 num_instances=1,
                 target_steps = 1_000_000,
                 training_interval = 25_000_000,
                 mmr_save_frequency = 50_000_000
                 ) -> None:
        self.frame_skip = frame_skip          # Number of ticks to repeat an action
        self.half_life_seconds = half_life_seconds   # Easier to conceptualize, after this many seconds the reward discount is 0.5

        self.fps = fps / self.frame_skip
        self.gamma = np.exp(np.log(0.5) / (self.fps * self.half_life_seconds))  # Quick mafs
        self.agents_per_match = agents_per_match
        self.num_instances = num_instances
        self.target_steps = target_steps
        self.steps = self.target_steps // (self.num_instances * self.agents_per_match) #making sure the experience counts line up properly
        self.batch_size = self.target_steps//10 #getting the batch size down to something more manageable - 100k in this case
        self.training_interval = training_interval
        self.mmr_save_frequency = mmr_save_frequency

        #the expected location of TASystemSettings.ini (default: %USERPROFILE%\Documents\my games\Rocket League\TAGame\Config)
        self.rl_config_ini = os.environ['USERPROFILE']+"\\Documents\\my games\\Rocket League\\TAGame\\Config\\TASystemSettings.ini"
        #the ini file best used for training (small resolution, no rendering)
        self.rl_training_ini = "./TASystemSettings_Training.ini"
        #replace the file below with a backup of YOUR TASystemSettings.ini
        self.rl_original_ini = "./TASystemSettings_Orig.ini"

        #some debug/info output, can be commented out
        print(f"Training Interval: {format(self.training_interval, ',d')}")
        print(f"Target Steps: {format(self.target_steps, ',d')}")
        print(f"Steps: {format(self.steps, ',d')}")
        print(f"Batchsize: {format(self.batch_size, ',d')}")
        print(f"mmr_save_frequency: {format(self.mmr_save_frequency, ',d')}")
        print(f"agents_per_match: {format(self.agents_per_match, ',d')}")
        print(f"num_instances: {format(self.num_instances, ',d')}")

    #saves the current model
    def exit_save(self, model) -> None:
       self. model.save("models/exit_save")

    #return an rlygm match instance
    def get_match(self) -> Match:  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=self.agents_per_match // 2,
            tick_skip=self.frame_skip,
            spawn_opponents=True,
            reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0,
                    concede=-100.0,
                    shot=5.0,
                    save=30.0,
                    demo=10.0,
                ),
            ),
            (0.1, 1.0, 1.0)),
            terminal_conditions=[TimeoutCondition(self.fps * 300), NoTouchTimeoutCondition(self.fps * 45), GoalScoredCondition()],
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    #tries to load an existing model and creates a new one if no existing model found
    def load_model(self) -> None:

        self.env = SB3MultipleInstanceEnv(self.get_match, self.num_instances)            # Start 1 instances, waiting 60 seconds between each
        self.env = VecCheckNan(self.env)                                # Optional
        self.env = VecMonitor(self.env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
        self.env = VecNormalize(self.env, norm_obs=False, gamma=self.gamma)  # Highly recommended, normalizes rewards

        #try to load an existing model
        try:
            self.model = PPO.load(
                "models/exit_save.zip",
                self.env,
                device="auto",
                #custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
                # If you need to adjust parameters mid training, you can use the below example as a guide
                custom_objects={"n_envs": self.env.num_envs, "n_steps": self.steps, "batch_size": self.batch_size, "n_epochs": 10, "learning_rate": 5e-5}
            )
            print("Loaded previous exit save.")
        
        #no existing model found
        except: 
            print("No saved model found, creating new model.")
            from torch.nn import Tanh
            policy_kwargs = dict(
                activation_fn=Tanh,
                net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
            )

            self.model = PPO(
                MlpPolicy,
                self.env,
                n_epochs=10,                 # PPO calls for multiple epochs
                policy_kwargs=policy_kwargs, # custom 
                learning_rate=5e-5,          # Around this is fairly common for PPO
                ent_coef=0.01,               # From PPO Atari
                vf_coef=1.,                  # From PPO Atari
                gamma=self.gamma,            # Gamma as calculated using half-life
                verbose=3,                   # Print out all the info as we're going
                batch_size=self.batch_size,  # Batch size as high as possible within reason
                n_steps=self.steps,          # Number of steps to perform before optimizing network
                tensorboard_log="logs",      # `tensorboard --logdir out/logs` in terminal to see graphs
                device="auto"                # Uses GPU if available
            )

    #runs the actual training
    def train(self) -> None: # Save model every so often
        # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
        # This saves to specified folder with a specified name
        self.callback = CheckpointCallback(round(5_000_000 / self.env.num_envs), save_path="models", name_prefix="rl_model")

        try:
            self.mmr_model_target_count = self.model.num_timesteps + self.mmr_save_frequency
            while True:
                #may need to reset timesteps when you're running a different number of instances than when you saved the model
                self.model.learn(self.training_interval, callback=self.callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
                self.model.save("models/exit_save")
                if self.model.num_timesteps >= mmr_model_target_count:
                    self.model.save(f"mmr_models/{model.num_timesteps}")
                    self.mmr_model_target_count += self.mmr_save_frequency

        except KeyboardInterrupt:
            print("Exiting training")

        print("Saving model")
        self.exit_save(self.model)
        self.restore_rl_config()
        print("Save complete")

    #copies ini file optimized for training
    def prepare_rl_config(self) -> None :
        print(f"Copying TRAINING INI RL config folder")
        try:
            shutil.copyfile(self.rl_training_ini, self.rl_config_ini)
        except:
            print(f"Error copying TRAINING INI")
            
    #restores original ini file for gaming
    def restore_rl_config(self) -> None :
        print(f"Restoring ORIGINAL INI...") 
        try:
            shutil.copyfile(self.rl_original_ini, self.rl_config_ini)
        except:
            print(f"Error restoring ORIGINAL INI File")



if __name__ == '__main__':
    example_bot = YoutubeExampleBot(mmr_save_frequency = 50_000_000,  # make a save of the model every X steps for MMR comparison
                                    training_interval=25_000_000,     # the total steps of one training unit
                                    target_steps=1_000_000,           # number of steps before re-training the model
                                    agents_per_match=6,               # Agents per match (3v3: 6, 2v2: 4, 1v1: 2)
                                    num_instances=1)                  # number of RL instances (as much as your PC can handle)
    
    example_bot.prepare_rl_config() #copy ini file optimized for training
    example_bot.load_model()
    example_bot.train()
    example_bot.restore_rl_config() #restore backup ini file for gaming
import random
from collections import deque
import numpy as np
import torch
import torch.onnx
import torch.nn.functional as F
from torch import nn, optim
import logging

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.torch_policy import TorchPolicy 
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer 
from mlagents.trainers.trajectory import Trajectory
from mlagents_envs.timers import hierarchical_timer
from mlagents_envs.base_env import ActionTuple

logger = get_logger(__name__)
TRAINER_NAME = "custom_dqn"

logging.basicConfig(level=logging.DEBUG)
"""The CustomDQNSetting class extend OffPolicyHyperparamSettings, which provides some attribute
 for the implementation of Off Policy Trainers. There are extra hyperparameters like epsilon values and gamma
 but the values assigned below are overwritten by the full set of hyperparameters present in the 
 YAML file at the following path: customTrainer/config/agent_ball_config.yml """
import attr    
@attr.s(auto_attribs=True)
class CustomDQNSettings(OffPolicyHyperparamSettings):
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    gamma: float = 0.99
   
""" The CustomDQNTrainer extends OffPolicyTrainer which is initialized once the ml-agents trainer is started.
When this happen, the create_policy and create_optimizer are called to create a self.policy and self.optimizer objects"""    
class CustomDQNTrainer(OffPolicyTrainer):
    def __init__(self, 
                 behavior_name: str, 
                 reward_buff_cap: int, 
                 trainer_settings: TrainerSettings, 
                 training: bool, 
                 load: bool, 
                 seed: int, 
                 artifact_path: str):
        logging.debug(f"Initializing {TRAINER_NAME} with settings: {trainer_settings}")
        super().__init__(behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path)
        self.policy = None
        self.target_policy = None  
        self.replay_buffer = ReplayBuffer(trainer_settings.hyperparameters.buffer_size)
        self.epsilon = trainer_settings.hyperparameters.epsilon_start
        self.epsilon_end = trainer_settings.hyperparameters.epsilon_end
        self.epsilon_decay = trainer_settings.hyperparameters.epsilon_decay
        self.batch_size = trainer_settings.hyperparameters.batch_size
        self.gamma = trainer_settings.hyperparameters.gamma
        self.target_update_interval = 1000  
        self.summary_freq = 10000
        self.step_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.completed_episodes = 0 
        self.group_episode_rewards = []
        self.current_group_episode_reward = 0
    
    
    def create_policy(self, parsed_behavior_id, behavior_spec ):
        return DQNPolicy(
            self.seed, behavior_spec, self.trainer_settings, self.epsilon, DQNetwork, {}
        )
    
    def create_optimizer(self):
        return DQNOptimizer(
            self.policy, self.trainer_settings
        )
    
    
    """ The _process_trajectory method manage the interaction with the environment,
    collect experiences and memorize them in the replay buffer. Every experience includes observations, actions, rewards, next state.
    When calling forward on a policy, the argument will include an “experience” dictionary from the last step.
    The forward method will generate an action and the next “experience” dictionary.
    Here, when the replay buffer contains enough experience, is called the update method of the optimizer and periodically 
    the target network is udated"""
    
    @hierarchical_timer("process_trajectory")
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id
        for idx, step in enumerate(trajectory.steps):
            experience = {
                'observations': step.obs,
                'actions': step.action,
                'rewards': step.reward,
                'done': step.done,
                'interrupted': step.interrupted,
                'next_observations': trajectory.next_obs,
                'group_rewards': step.group_reward
            }
            
            self.replay_buffer.add(experience)
            self.step_count += 1 
            
            if self.step_count % self.summary_freq == 0:
                self._log_summary_statistics()

            self.current_episode_reward += step.reward
            self.current_group_episode_reward += step.group_reward

            if step.done or step.interrupted:
                self.episode_rewards.append(self.current_episode_reward)
                self.group_episode_rewards.append(self.current_group_episode_reward)
                self.current_episode_reward = 0
                self.completed_episodes += 1
                self._stats_reporter.add_stat("Completed Episodes", self.completed_episodes)
        # If enough samples in buffer, update the policy
        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            update_stats =self.optimizer.update(batch, num_sequences=None)
            for stat_name, stat_value in update_stats.items():
                self._stats_reporter.add_stat(stat_name, stat_value)
            self.policy.update_epsilon()

            # Periodically update the target network
            if self.step_count % self.target_update_interval == 0:
                logging.debug("Updating target network")
                self.optimizer.target_policy.load_state_dict(self.policy.network.state_dict())
        
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

        
    def _log_summary_statistics(self):
        # Calculate and log mean episode reward
        if self.episode_rewards:
            mean_episode_reward = np.mean(self.episode_rewards)
            std_episode_reward = np.std(self.episode_rewards)
            mean_group_episode_reward = np.mean(self.group_episode_rewards)
            std_group_episode_reward = np.std(self.group_episode_rewards)

            self._stats_reporter.add_stat("Mean Episode Reward", mean_episode_reward)
            self._stats_reporter.add_stat("Std Episode Reward", std_episode_reward)
            self._stats_reporter.add_stat("Mean Group Episode Reward", mean_group_episode_reward)
            self._stats_reporter.add_stat("Std Group Episode Reward", std_group_episode_reward)
            self._stats_reporter.add_stat("Completed Episodes", self.completed_episodes)
            logger.info(f"Step: {self.step_count}. Mean Episode Reward: {mean_episode_reward:.5f}.  Mean Group Episode Reward: {mean_group_episode_reward:.2f}. Completed Episodes: {self.completed_episodes}.")
            # Reset episode rewards
            self.episode_rewards = []
            self.group_episode_rewards = []


    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    
def get_type_and_setting() :
    logging.debug("Registering CustomDQNTrainer")
    trainer_mapping = {CustomDQNTrainer.get_trainer_name(): CustomDQNTrainer}
    settings_mapping = {CustomDQNTrainer.get_trainer_name(): CustomDQNSettings}
    logging.debug(f"Trainer mapping: {trainer_mapping}")
    logging.debug(f"Settings mapping: {settings_mapping}")
    return trainer_mapping, settings_mapping

#Replay Buffer to collect experiences
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Architecture of the Q Network and target network
class DQNetwork(nn.Module):
    def __init__(self,  observation_specs, network_settings, action_spec):
        super(DQNetwork, self).__init__()
        input_dim = observation_specs[0].shape[0]
        #logging.debug(f"input_dim: {input_dim}")
        output_dim = action_spec.continuous_size
        hidden_units = network_settings.hidden_units

        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)
        self.memory_size = 0

    def forward(self, observations, network_settings, action_spec):
        if not isinstance(observations, torch.Tensor):
            observations = observations[0]
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""At each training step the evaluate method of the policy is called to take a random action or an action
returned by the q network, accordingly to the epsilon-greedy strategy.
Then the action is taken and the experience informations are processed in _process_trajectory()"""
class DQNPolicy(TorchPolicy):
    def __init__(self, seed, behavior_spec, settings, epsilon, actor_cls, actor_kwargs):
        super().__init__(seed, behavior_spec, settings.network_settings, actor_cls, actor_kwargs)
        self.network = actor_cls(
            observation_specs=behavior_spec.observation_specs,
            network_settings=settings.network_settings,
            action_spec=behavior_spec.action_spec,
            **actor_kwargs
        )
        self.settings = settings
        self.optimizer = optim.Adam(self.network.parameters(), lr=settings.hyperparameters.learning_rate)
        self.epsilon = epsilon 
        
    def evaluate(self, decision_requests, global_agent_ids) :
        observations = decision_requests.obs
        num_agents = len(observations[0])
        if random.random() < self.epsilon:
            actions = [random.choice(range(self.behavior_spec.action_spec.discrete_branches[0])) for _ in range(num_agents)]
            continuous_actions = np.random.uniform(-1, 1, size=(num_agents, self.behavior_spec.action_spec.continuous_size))
        else:
            observations = torch.tensor(observations[0], dtype=torch.float32)
            with torch.no_grad():
                actions = self.network(observations, self.network_settings, self.behavior_spec.action_spec).argmax(dim=1).tolist()
                continuous_actions = self.network(observations, self.network_settings, self.behavior_spec.action_spec).numpy()
                continuous_actions = continuous_actions.reshape(num_agents, self.behavior_spec.action_spec.continuous_size)    
        run_out = {
            "action": ActionTuple(discrete=np.array(actions).reshape(2,1), continuous=continuous_actions),
            "env_action": ActionTuple(discrete=np.array(actions).reshape(2,1), continuous=continuous_actions),
        } 
        return run_out
    
    def update_epsilon(self):
        if self.epsilon > self.settings.hyperparameters.epsilon_end:
            self.epsilon *= self.settings.hyperparameters.epsilon_decay
    

class DQNOptimizer(TorchOptimizer):
    def __init__(self, policy, trainer_settings):
        super().__init__(policy, trainer_settings)
        self.policy = policy
        self.trainer_settings = trainer_settings
        # Initializing Target Policy here
        self.target_policy = DQNetwork(
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=policy.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        
    @property
    def critic(self):
        return self.target_policy
    
    def update(self, batch, num_sequences):
        observations = torch.tensor(np.array([exp['observations'] for exp in batch]), dtype=torch.float32).squeeze(1)
        #actions_discrete = torch.tensor(np.array([exp['actions'].discrete for exp in batch]), dtype=torch.int64).unsqueeze(1)
        actions_continuous = torch.tensor(np.array([exp['actions'].continuous for exp in batch]), dtype=torch.float32)
        rewards = torch.tensor(np.array([exp['rewards'] for exp in batch]), dtype=torch.float32)
        next_observations = torch.tensor(np.array([exp['next_observations'] for exp in batch]), dtype=torch.float32).squeeze(1)
        dones = torch.tensor(np.array([exp['done'] for exp in batch]), dtype=torch.float32)
        group_rewards = torch.tensor(np.array([exp['group_rewards'] for exp in batch]), dtype=torch.float32)

        # Consider both individual and group rewards
        total_rewards = rewards + group_rewards
        
        # Compute Q-values for the current observations and actions
        q_values = self.policy.network(observations, self.policy.network_settings, self.policy.behavior_spec.action_spec)
        actions_indices = actions_continuous.argmax(dim=1, keepdim=True)
        q_values = q_values.gather(1, actions_indices).squeeze()

        # Compute Q-values for the next observations
        with torch.no_grad():
            next_q_values = self.target_policy(next_observations, self.policy.network_settings, self.policy.behavior_spec.action_spec).max(dim=1)[0]
        
        # Compute target Q-values
        target_q_values = total_rewards + self.trainer_settings.hyperparameters.gamma * next_q_values * (1 - dones)
        
        loss = F.mse_loss(q_values, target_q_values.detach())
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        # Collect statistics
        update_stats = {
            "Losses/Value Loss": loss.item(),
            "Policy/Epsilon": self.policy.epsilon,
            "Rewards/Total Rewards": total_rewards.mean().item(),
            "Q Values/Current Q Values": q_values.mean().item(),
            "Q Values/Target Q Values": target_q_values.mean().item()
        }

        return update_stats

    def get_modules(self):
        return {"policy": self.policy.network, "optimizer": self.policy.optimizer}






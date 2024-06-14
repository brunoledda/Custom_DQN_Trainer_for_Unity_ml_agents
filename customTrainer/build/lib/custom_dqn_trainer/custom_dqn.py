import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings, ScheduleType #Vedere meglio
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.torch_policy import TorchPolicy #Vedere meglio
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer #Vedere meglio
from mlagents.trainers.trajectory import Trajectory
from mlagents_envs.timers import hierarchical_timer
#from typing import Dict

from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic #Vedere meglio

TRAINER_NAME = "custom_dqn"
class CustomDQNTrainer(OffPolicyTrainer):
    def __init__(self, trainer_settings: TrainerSettings, **kwargs):
        super().__init__(trainer_settings, **kwargs)
        self.policy = self.create_policy()
        self.target_policy = self.create_policy()  # For target network 
        #self.optimizer = self.create_optimizer() we are using adam optimizer instead of a customization
        self.replay_buffer = ReplayBuffer(trainer_settings.hyperparameters['buffer_size'])
        self.epsilon = trainer_settings.hyperparameters['epsilon_start']
        self.epsilon_end = trainer_settings.hyperparameters['epsilon_end']
        self.epsilon_decay = trainer_settings.hyperparameters['epsilon_decay']
        self.batch_size = trainer_settings.hyperparameters['batch_size']
        self.gamma = trainer_settings.hyperparameters['gamma']
        self.target_update_interval = 1000  # Aggiornamento della rete target ogni 1000 passi

    """ def create_policy(self) -> TorchPolicy:
        # Define your policy architecture here
        reward_signal_configs = self.trainer_settings.reward_signals
        reward_signal_names = [
            key.value for key, _ in reward_signal_configs.items()
        ]
        actor_cls = SharedActorCritic
        actor_kwargs = {"conditional_sigma": False, "tanh_squash": False, "stream_names": reward_signal_names}
        policy = TorchPolicy(
            self.seed,
            self.behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs
        )
        return policy """
    
    def create_policy(self):
        return DQNPolicy(
            self, self.behavior_spec, self.seed, self.trainer_settings,
            actor_cls=DQNetwork, actor_kwargs={}
        )

    """ def create_optimizer(self) -> TorchOptimizer:
        return TorchOptimizer(
            self.policy, self.trainer_settings
        ) """
    
    """ @hierarchical_timer("process_trajectory")
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()
        #TO ADAPT FOR MY SCENARIO

        value_estimates, value_next, value_memories = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(v)
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )
        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        self._append_to_update_buffer(agent_buffer_trajectory) """
    
    """ A trajectory will be a list of dictionaries of strings mapped to Anything.
        When calling forward on a policy, the argument will include an “experience” dictionary from the last step.
        The forward method will generate an action and the next “experience” dictionary.
        Examples of fields in the “experience” dictionary include observation, action, reward, done status,
        group_reward, LSTM memory state, etc. """
    
    @hierarchical_timer("process_trajectory")
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        super()._process_trajectory(trajectory)
        for step in trajectory.steps:
            experience = {
                'observations': step.obs,
                'actions': step.action,
                'rewards': step.reward,
                'next_observations': step.next_obs,
                'dones': step.done
            }
            self.replay_buffer.add(experience)

    @hierarchical_timer("update_policy")
    def _update_policy(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        self.policy.update(batch)
        self.update_epsilon()

        # Aggiorna la rete target periodicamente
        if self.step_count % self.target_update_interval == 0:
            self.target_policy.network.load_state_dict(self.policy.network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay


    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
    
def get_type_and_setting() :
    return {CustomDQNTrainer.get_trainer_name: CustomDQNTrainer},{CustomDQNTrainer.get_trainer_name: DQNSettings}

# Definizione del Replay Buffer per memorizzare le esperienze
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Definizione della rete neurale DQN
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Definizione della Policy DQN
class DQNPolicy(TorchPolicy):
    def __init__(self, trainer, behavior_spec, seed, settings, actor_cls, actor_kwargs):
        super().__init__(trainer, behavior_spec, seed, settings, actor_cls, actor_kwargs)
        self.network = DQNetwork(
            behavior_spec.observation_specs[0].shape[0],
            behavior_spec.action_spec.discrete_branches[0],
            settings.network_settings.hidden_units
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=settings.hyperparameters['learning_rate'])

    def evaluate(self, observations) :
        if random.random() < self.trainer.epsilon:
            action = random.choice(range(self.behavior_spec.action_spec.discrete_branches[0]))
        else:
            observations = torch.tensor(observations, dtype=torch.float32)
            with torch.no_grad():
                action = self.network(observations).argmax(dim=1).item()
        return {"action": action}

    def update(self, batch):
        observations = torch.tensor([exp['observations'] for exp in batch], dtype=torch.float32)
        actions = torch.tensor([exp['actions'] for exp in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([exp['rewards'] for exp in batch], dtype=torch.float32)
        next_observations = torch.tensor([exp['next_observations'] for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp['dones'] for exp in batch], dtype=torch.float32)

        q_values = self.network(observations).gather(1, actions).squeeze(1)
        next_q_values = self.network(next_observations).max(dim=1)[0]
        target_q_values = rewards + self.trainer.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

#CONTROLLARE SE HA SENSO
import attr    
@attr.s(auto_attribs=True)
class DQNSettings(OffPolicyHyperparamSettings):
    gamma: float = 0.99
    exploration_schedule: ScheduleType = ScheduleType.LINEAR
    exploration_initial_eps: float = 0.1
    exploration_final_eps: float = 0.05
    target_update_interval: int = 10000
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update
    
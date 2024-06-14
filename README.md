# Custom_DQN_Trainer_for_Unity_ml_agents
This project provides a custom plugin for the ml-agents toolkit, that implement Deep Q Network Reinforcement Learning algorithm evaluated in a multi-agent scenario.
***
You will find the main code for the agents implementation (like the observations vector, actions and rewards specifications) in the **./CooperativeAgentBall/Assets/Scripts** path, more precisely in:
- RollerAgent.cs
- TrainingAreaController.cs

The main code for the custom trainer and the DQN algorithm implementation is available in a single python file at **./customTrainer/custom_dqn_trainer/custom_dqn.py**, including comments to explain how it works.
***
Requirements: 
- Unity 2022.3.20f1
- ml-agents Release 21, please install it following the official documentation at https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md
***
To train a model with my custom DQN trainer follow these instructions:
- Open the unity editor by double clicking the Unity file at the following path: ./CooperativeAgentBall/Assets/Scenes/SampleScene
- After the installation of ml-agents you should have created a python **venv**
- to integrate my custom plugin, activate the **venv**, then navigate to the customTrainer folder of this repository
- execute the following command : **pip install -e .**
- start the training with **mlagents-learn config/agent_ball_config.yaml --run-id=<your_execution_id>**

***
References:
- https://unity-technologies.github.io/ml-agents/ML-Agents-Overview/
- https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Tutorial-Custom-Trainer-Plugin.md/.

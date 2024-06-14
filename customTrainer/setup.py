from setuptools import setup
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE
setup(
    name='custom_dqn',
    version='0.2',
    packages=['custom_dqn_trainer'],
    entry_points={
        ML_AGENTS_TRAINER_TYPE: [
            'custom_dqn=custom_dqn_trainer.custom_dqn:get_type_and_setting',
        ],
    },
    install_requires=[
        'mlagents',
        'torch',
        'numpy',
    ],
) 
""" setup(
    name='custom_dqn_trainer',
    version='0.2',
    packages=['custom_dqn_trainer'],
    entry_points={
        'mlagents.trainers': [
            'custom_dqn=custom_dqn_trainer.custom_dqn:get_type_and_setting',
        ],
    },
    install_requires=[
        'mlagents',
        'torch',
        'numpy',
    ],
) """
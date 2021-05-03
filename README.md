# Automated_Arftificial_Pancreas_Controller
This repository contains the codebase of our project for Reinforcement Learning (CSCE 689) 


# Contributors
Sudip Paul
Projna Paromita
Anurag Das

Usage
============
To install the necessary packages, run the following command:-
`pip install -r requirements.txt`
Please use python 3

To train a model,
`python ACKTR_train.py -g child -r default`
g represents a group, r refers to the reward function. You can also train the model on data from adults or adolescents.

Supported reward functions:-
- Default
- Magni Reward
- Cameron Reward
- Reward Target
- Risk Diff


Base Code
==========
[https://github.com/jxx123/simglucose](https://github.com/jxx123/simglucose)

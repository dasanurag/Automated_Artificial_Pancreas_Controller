# Automated_Arftificial_Pancreas_Controller
This repository contains the codebase of our automated artificial pancreas controller project for Reinforcement Learning (CSCE 689) 

Project Report
===========
[https://drive.google.com/file/d/1NF2-H5fGxqc8IxsVThwfp_MdxLeD5IFN/view?usp=sharing](https://drive.google.com/file/d/1NF2-H5fGxqc8IxsVThwfp_MdxLeD5IFN/view?usp=sharing)

Slides
========
[https://drive.google.com/file/d/1rKY3VlANGWD1iEt-YFXfr_pmaEA4gWIp/view?usp=sharing](https://drive.google.com/file/d/1rKY3VlANGWD1iEt-YFXfr_pmaEA4gWIp/view?usp=sharing)

Demo Link
==========
[https://drive.google.com/file/d/1HYExmBszsDEkmc2K0DPLY3QcIE4DnbfJ/view?usp=sharing](https://drive.google.com/file/d/1HYExmBszsDEkmc2K0DPLY3QcIE4DnbfJ/view?usp=sharing)

# Contributors
Sudip Paul <br/>
Projna Paromita <br/>
Anurag Das

Usage
============
To install the necessary packages, run the following command:-<br/>
`pip install -r requirements.txt`<br/>

To train a model,<br/>
`python ACKTR_train.py -g child -r default`<br/>
g represents a group, r refers to the reward function. You can also train the model on data from adults or adolescents.

Supported reward functions:-
- Default (Risk Diff)
- Magni Reward
- Cameron Reward
- Reward Target
- Risk Event

To evaluate a model on data from adult patients, <br/>
`python apply_customized_controller.py -g child -r default`<br/>

A number of pre-trained models are available under the "Saved_models" directory

Base Code
==========
[https://github.com/jxx123/simglucose](https://github.com/jxx123/simglucose)

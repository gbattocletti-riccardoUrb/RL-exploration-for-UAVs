# Reinforcement Learning-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments

This repository contains a collection of the files produced during the Master Thesis in Mechatronic Engineering we pursued in 2021 at Politecnico di Torino (Italy). The title of the thesis is "Reinforcement Learning-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments". A good overview of the project can be found in the paper we presented at the 2021 AIAA forum. In the paper, the main idea behind the project is explained, along with some technical detail about the development and implementation of the algorithm. Moreover, all the relevant results are presented and discussed. The paper is available on ResearchGate at https://www.researchgate.net/publication/353530217_RL-based_Path_Planning_for_Autonomous_Aerial_Vehicles_in_Unknown_Environments. The files collected here regard the simulation environment creation, the RL agents training and the simulation of the UAVs operations used to produce all the numeric results. Below, a brief description of each folder role inside the project can be found.

## Project description

The main goal of the project is to use Reinforcement Learning to produce an exploration algorithm capable of driving a fleet of UAVs in the exploration of an unknown environment.



## Repository folders

Each of the folders in this repository corresponds to a different step in the development process of the exploration algorithm.
- models: the folder contains the data of the trained models. Currently, in the folder are uploaded only three models (two for the path planning agent and one for the coverage one), which are amongst the best ones obtained during the training phase;
- maps: contains the map datasets used for the training and validation of the RL models. The folder contains several training datasets, which have been built over time to try and find the optimal map type in order to obtain a faster and more reliable training process. A set of validation maps is also present. The folder also contains the MATLAB files used to generate the maps. They can be used to generate new training and validation/simulation maps;
- media:
- main:
- main [only coverage]:
- main [only path planning]:
- main [3D]:
- training coverage:
- training path planning:

It is worth mentioning that, during the training process, we used the website *wandb* (aka "Weight and Biases", https://wandb.ai/) to log some useful training information. Moreover, we use the HPC cluster of Politecnico di Torino (https://hpc.polito.it/) to perform the trainings. Therefore, it could be necessary to comment or modify some lines in the main files in order to be able to succesfully start the trainings. In the same way, it could be necessary to modify the path to some folder (e.g. the training map folder) to match the actual location of the dataset on your PC.

## Contacts

For any question or suggestion feel free to contact us at battocletti.gianpietro@gmail.com or riccardourban@hotmail.it. Here on GitHub you can also visit our personal profiles https://github.com/gbattocletti and https://github.com/RiccardoUrb.

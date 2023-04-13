# Reinforcement Learning-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments

The repository contains the code developed for our thesis project in 2021 while pursuing the MSc in Mechatronic Engineering at Politecnico di Torino (Italy). The title of the thesis is 'Reinforcement Learning-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments'. The project regards the coordination of a fleet of UAVs for the cooperative exploration of indoor environments.

The complete thesis in which this project is described can be found at [this link](https://webthesis.biblio.polito.it/secure/19281/1/tesi.pdf). A paper with the main results from this work sa presented at the 2021 AIAA Aviation Forum. The paper can be read on the [AIAA conference website](https://arc.aiaa.org/doi/pdf/10.2514/6.2021-3016) or on [ResearchGate](https://www.researchgate.net/publication/353530217_RL-based_Path_Planning_for_Autonomous_Aerial_Vehicles_in_Unknown_Environments).

If you use or modify parts of this project in your work, please cite it as:

_G. Battocletti, R. Urban, S. Godio and G. Guglieri, "RL-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments," AIAA AVIATION 2021 FORUM, 2021._ 

Or using the follwing bibtex entry:

	@inbook{battocletti2021rl,
		title={RL-based Path Planning for Autonomous Aerial Vehicles in Unknown Environments},
		author={Battocletti, Gianpietro and Urban, Riccardo and Godio, Simone and Guglieri, Giorgio},
		booktitle={AIAA AVIATION 2021 FORUM},
		year={2021},
		doi={https://doi.org/10.2514/6.2021-3016}
	}  
	
## Project description

The main goal of the project is to use Reinforcement Learning (RL) to implement an *exploration algorithm* capable of driving a small fleet of UAVs in the exploration of an unknown environment. This kind of task presents several different challenges. In fact, each drone must be capable of moving in space without hitting any obstacle or other drones. At the same time, it has to continue the exploration task - or any other task assigned to it. While performing these tasks, the drones must also communicate with each other in order to coordinate the exploration following a common strategy and share useful information to optimise the execution of the task. All these issues have to be solved and their solutions merged in an organic algorithm. The focus of the project is on the sections of algorithm regarding path planning, obstacle avoidance and exploration coordination. 

The case study scenario assumes that a number *n* of UAVs is placed in some given initial locations inside of an unknown region. The task of the UAV fleet is to explore (i.e. obesrve through the use of sensors and cameras) as much as possible of the environment in the least amount of time. The approach taken during the development of the exploration algorithm has been to split the exploration task in two parts. The first part is called *coverage* algorithm and has the task of coordinating the exploration process. A copy of this algorithm runs on each of the UAVs; each UAV is responsible for the decision of its own target location. Each UAV computes its own target point by taking into account the avaliable data about environment shape, obstacle position and other UAVs location. By accessing these info, each UAV computes its own target location by running the *coverage agent*, i.e. a properly RL-trained Neural Network (NN). The image below shows how a fleet of 4 UAVs exploit the coverage agent to generate 4 different target locations.

<p align="center">
	<img src="/media/coverage_output.png" alt="Coverage NN output" width="500"/>
</p>
 
Once each UAV has its target point, the second piece of algorithm, the *path planning algorithm*, is called into action. The path planning algorithm is composed by a second RL-trained Neural Network called *path planning agent*. This second agent is able to compute a suitable flight trajectory to lead the UAV to its target location avoiding obstacles and other UAVs. The path planning agnet works in real time so that obstacles that are detected during flight can be immediately taken into account to update the trajectory and avoid collisions with them. An example of trajectory planning is shown below (red line is the trajectory, which starts from the red dot which represents one UAV location. Black squares are known obstacles and grey ones are unknown obstacles).

<p align="center">
	<img src="/media/sim1_1.png" alt="Path Planning trajectory" width="300"/>
</p>

THe exploration algorithm works by continuously calling in action the two agents described above. The symultaneous use of the two pieces of algorithm result in an effective exploration algorithm, as can be seen in the the simulations in the *media* folder and in the numerical results discussed in the paper.

## Repository folders

This repository contains several folder, each one corresponding to a different development stage of the algorithm:
- **main:** main simulation of the complete exploration algorithm. The main file allows to select the desired map, the path planning and coverage agents, and some other parameters;
- **main [only coverage]:** simplified simulation in which the focus is on the coverage agent. The path planning agent is replaced by a simpler (and less efficient) path planning algorithm. This allows to speed up the simulation runtime and focus on the evaluation of the coverage performances;
- **main [only path planning]:** simplified simulation in which the focus is on the path planning agent. In this case, the coverage agent is replaced by a random placement (or manual placement) of the target points of each UAV. This way, the focus is on the path planning agent behaviour and on the obstacle avoidance effectiveness;
- **main [3D]:** this folders contains a first implementation of the algorithm in a 3D scenario. All the previous simulations are run in a 2D environment. In this folder, a simplified approach to the exploration of a 3D environment is implemented and analysed. The approach used is to "slice" the environment along the *z* axis and place one UAV on each height level. Each UAV only moves at a fixed height, behaving as if it was in a 2D environment. This approach is clearly sub-optimal, but allows to easily extend the 2D algorithm to a real-world scenario. An example of 3D exploration is shown below.

<p align="center">
	<img src="/media/3D_sim.png" alt="3D simulation with height slicing" width="500"/>
</p>

- **training path planning:** this folder contains the files containing the tensorflow code used to train the path planning agent. This includes the Neural Network architecture definition, functions to manage state and reward, and the routine to perform the SGD and backprop operations;
- **training coverage:** same as above but for the coverage agent. In this case, the issue is that the training is very slow and should be improved/optimised. At the moment, in fact, the amount of time to reach a reasonable number of training episodes is *very* large (many days of training). Future work on this project should focus on this section. In addition to the speed up required by the coverage agent training, it could be necessary to  review the coverage agent structure and training strategy;
- **models:** the folder contains the trained models (i.e., trained RL agents). Currently, in the folder are uploaded only a few models, which are some of the best ones obtained during the training phase. These models are the ones that can be used during the simulations in the *main* folders;
- **maps:** the folder contains the map datasets used for the training and validation of the algorithm. The folder contains several training datasets, which have been built over time to try and find the optimal map type to obtain a faster and more reliable training process. A set of validation maps is also present. The folder also contains the MATLAB files used to generate the maps. They can be used to generate new training and validation/simulation maps;
- **media:** lastly, the media folder contains some images and videos that show the results obtained. Some more images and plots (which illustrate the performance quality of the algorithm) can be found in the aforementioned paper on ResearchGate.

## Notes

During the development of the project, and in particular in the NNs training process, we used the website Weight and Biases (https://wandb.ai/) to log useful training information, and the HPC cluster of Politecnico di Torino (https://hpc.polito.it/) to obtain the computational power required to perform the trainings. Therefore, it could be necessary to comment or modify some lines in the main files in order to be able to succesfully start the trainings. For the same reason, it may be necessary to modify the path to some folder (e.g. the training maps folder) to match the actual location of the directories on your PC.

Another useful link is this [YouTube tutorial](https://www.youtube.com/watch?v=KZFn0dvPZUQ) on the setup of the tensorflow environment (with support for GPU cuda) that is used during the training of the planning agents.

## Contacts

For any question or suggestion feel free to contact us at battocletti.gianpietro@gmail.com or riccardourban@hotmail.it.

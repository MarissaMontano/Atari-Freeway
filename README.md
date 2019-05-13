




________
|  ______|  ______    _____     _____  ___        ___            ____  ____
| |___    _/  ____|  / /_\ \   / /_\ \ \  \  /\  /  /  ______    \   \/   /
|  ___|   |  |      |  ____/  |  ____/  \  \/  \/  /  /   _   \   \_    _/
| |       |  |      |  \____  |  \____   \   /\   /  |   |_|   \    |  |
|_|       |__|       \_____/   \_____/    \_/  \_/    \______/|_|   |__|

README

Reinforcement learning with Q-Learning for the game Freeway 
	This is a AI agent that is trained to learn the optimum policy of the Atari game Freeway using Q-learning. 
    This agent learns with a type of q-learning called epsilon/exploration rate Greedy Q-Learning. This means 
    it learns the optimum policy from exploration annealing. In the beginning it wants to explore everything, but 
    close to the end it should only exploit paths it already knows are safe. 


Installations/Dependencies
	- install Python >= 3.0
    - install Numpy
    - install Gym
    - install gym[atari]
      (for windows users, may need to look at https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299)

Code
	Our code was written in python using gym and the atari game freeway-ram-v0. To install dependencies, please follow online tutorials. 
	There are three important user configurations to our code. The first one is the variable episode which can be found at the bottom line 120. This is how many rounds the game will run (2 min and 16 seconds per round) 
	The next configuration comes on line 44. We currently have our environment to not show the game being played. This is because the graphics made the game take longer to load. For smaller trials, uncomment and adjust to see the trials
	Finally, on line 103, the user can get feedback for avg score at a certain episode. Feel free to adjust if you want to make sure code is running and see progress appropriately. 



How to run
    to run the program all you need to do is call "python freeway.py" in a terminal or run it from an IDE. 


Picture File: 
These contains the graphs for our project
	Rewards10
		Forward reward +1
	Rewards25
		Forward reward +1
	Rewards100
		Forward reward +1
	Rewards1000
		Forward reward +1
	Rewards100A
		Forward reward +1, Backward rewards -2
	Rewards100B
		Forward Reward +2, Backward Reward -2, No-op +1
	Reward1000B
		Extra test 100B rewards with 1000 runs

	
Credits 
	This project was made possible by Marissa Montano and Jorge Moreno

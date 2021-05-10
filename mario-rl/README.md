# Introduction
This project is made for Reasoning Agents course that will be having Mario character and applying Reinforcement Learning in order to train the model. 

## How to run
In order to run, you need to make several steps
1. Install Python
2. Install required libraries 
3. Move to virtual environment with
```
venv\Scripts\activate
```

## Details 
Everything is detailed in the ```main.ipynb``` file with the instructions. Firtly, we imported the libraries and made our environment for Mario GUI. Then, we created the class for the MarioAgent for the reinforcement learning part by including all the functions for building the model and training. Then, finally, we tested the trained model. As you see from the code, it trains the model when the mario.memory is more than batch size (batch size we gave was 64). After the model getting trained. We could save and load our model for the visualizing it. 
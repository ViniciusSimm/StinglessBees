-> architecture.py
This file contains all the various architectures utilized in this paper.
Additional models can be included and modified.
The "freeze" argument is responsible for disabling training on certain layers (a suggested but not mandatory value of 20 is used).

-> main.py
This file is responsible for training the model.
The model name ("MODEL") will distinguish all the saved files.
If a model already exists, the training will continue from its current state, reusing the existing weights.
Two callbacks are employed to save training history and checkpoints.
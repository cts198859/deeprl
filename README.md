# DeepRL
## Purpose
The implementation of deepRL agents for both discrete and continuous (only single dimension) controls. The inputs are time series measurements and the output is the control. 

## Setup
In your bashrc you need to have:
```
$ export DEEPCUT_SRC_DIR=/path/to/deepcut/src/code/dir
$ export DEEPCUT_RESULT_DIR=/path/to/results/dir

$ export RL_SRC_DIR=/path/to/deepcut/src/code/dir
$ export RL_RESULT_DIR=/path/to/results/dir

Typically DEEPCUT_RESULT_DIR and RL_RESULT_DIR are the same!
```


## Usage
To use a Gym Env:

Define all parameters in `config.ini`, and run `python3 main.py --config-path [path to config.ini]`. This is in run.sh
Multi-processing implementation is at multiprocess branch, in which the global wt and local batch are maintained in queues. It is not as optimal as the multi-threading implementation due to the potential lag between the generation and consumpution of each local batch.

In config-path, the variable BASE DIR has the directory where results are present. Go there on your machine and it will have subfolders of log/, model/.
To monitor progress on tensorboard, type `python -m tensorflow.tensorboard --logdir=.` which will launch tensorboard and you can monitor progress on a browser window. Some example plots are below.

To use the Drone Env:
    In deepcut results dir, run:
        1. simple_rmse.py
        2. plot_generator.py

    To train RL for the drone problem, run:
        1. drone_run.sh 


## Example results
**continuous control**     | discrete control
:-------------------------:|:--------------------------:
Pendulum                   | Acrobot
![](./docs/pend.png)       | ![](./docs/acro.png)
MountainCarContinuous      | MountainCar
![](./docs/contcar.png)    | ![](./docs/car.png)



detailed config files are located under `./docs`.



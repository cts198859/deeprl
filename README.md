# DeepRL
## Purpose
The implementation of deepRL agents for both discrete and continuous (only single dimension) controls. The inputs are time series measurements and the output is the control. 

## Usage
Define all parameters in `config.ini`, and run `python3 main.py --config-path [path to config.ini]`. This is in run.sh
Multi-processing implementation is at multiprocess branch, in which the global wt and local batch are maintained in queues. It is not as optimal as the multi-threading implementation due to the potential lag between the generation and consumpution of each local batch.

In config-path, the variable BASE DIR has the directory where results are present. Go there on your machine and it will have subfolders of log/, model/.
To monitor progress on tensorboard, type `python -m tensorflow.tensorboard --logdir=.' which will launch tensorboard and you can monitor progress on a browser window. Some example plots are below.

python -m tensorboard.main --logdir=.


## Example results
**continuous control**     | discrete control
:-------------------------:|:--------------------------:
Pendulum                   | Acrobot
![](./docs/pend.png)       | ![](./docs/acro.png)
MountainCarContinuous      | MountainCar
![](./docs/contcar.png)    | ![](./docs/car.png)



detailed config files are located under `./docs`.



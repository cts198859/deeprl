# DeepRL
## Purpose
The implementation of deepRL agents for both discrete and continuous (only single dimension) controls. The inputs are time series measurements and the output is the control. 

## Usage
Define all parameters in `config.ini`, and run `python3 main.py --config-path [path to config.ini]`.

## Example results
continuous control                           | discrete control
:-------------------------------------------:|:-----------------------------------------------:
![Pendulum](./docs/pend.png)                 | ![Acrobot](./docs/acro.png)
:-------------------------------------------:|:------------------------------:
![MountainCarContinuous](./docs/contcar.png) | ![MountainCar](./docs/car.png)



detailed config files are located under `./docs`.

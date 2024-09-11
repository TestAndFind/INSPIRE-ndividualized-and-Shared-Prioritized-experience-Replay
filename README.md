# INSPIRE: Individualized and Shared Prioritized experience Replay for sparse reward multi-agent reinforcement learning
The code of INSPIRE: individualized and Shared Prioritized experience Replay for sparse reward multi-agent reinforcement learning(Still in testing and buliding))

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py --config=INSPIRE --env-config=sc2 with env_args.map_name=2s3z t_max=3005000

# Isaac-ManipulaRL

![Image text](img-folder/2222.png)

[![Author](https://img.shields.io/badge/Author-cypypccpy-blue.svg "Author")](https://github.com/cypypccpy "Author")
[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
<br></br>

## Pause development 
I am very sorry that the development of this project will be suspended for a period of time. But you can refer to my latest work with IsaacGym: 

We present the RL-based dual dexterous hand environment, Bi-DexHands, which provides a collection of bimanual dexterous manipulations tasks and reinforcement learning algorithms for solving them. Reaching human-level sophistication of hand dexterity and bimanual coordination remains an open challenge for modern robotics researchers.

Bi-DexHands contains complex dexterous hands control tasks. Bi-DexHands is built in the NVIDIA Isaac Gym with a high-performance guarantee for training RL algorithms. Our environments focus on applying model-free RL/MARL algorithms for bimanual dexterous manipulation, which are considered as a challenging task for traditional control methods.

Please visit the github page for more details and examples:
[https://github.com/PKU-MARL/DexterousHands/](https://github.com/PKU-MARL/DexterousHands/)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [TODO](#todo)
- [Contributing](#contributing)
- [License](#license)
<br></br>

## Background

The Manipulator Reinforcement Learning based on [Isaac-gym](https://developer.nvidia.com/isaac-gym), the following additional implementations are added:
- Add Baxter and UR5 robots and supporting environment like open cabinet, assembly and pick & place
- Customizable neural network structure
- Visual input reinforcement learning processing pipeline
- SAC2019 Algorithm
- Reinforcement Learning from Demonstration
- ROS wrapper

This project is **still under development** and detailed usage documentation will be available upon completion.
<br></br>

## Install
### Prerequisites
- Ubuntu 18.04 or 20.04.
- Python 3.6, 3.7 or 3.8.
- Minimum recommended NVIDIA driver version:

  + Linux: 460.32

### Set up the Python package

#### Install in an existing Python environment

In the ``python`` subdirectory, run:

    pip install -e .

This will install the ``isaacgym`` package and all of its dependencies in the active Python environment.  If your have more than one Python environment where you want to use Gym, you will need to run this command in each of them.  To verify the details of the installed package, run::

    pip show isaacgym

To uninstall, run::

    pip uninstall isaacgym

#### Install in a new conda environment

In the root directory, run:

    ./create_conda_env_rlgpu.sh

This will create a new conda env called ``rlgpu``, which you can activate by running:

    conda activate rlgpu

If you wish to change the name of the env, you can edit ``python/rlgpu_conda_env.yml``, then update the ``ENV_NAME`` variable in the ``create_conda_env_rlgpu.sh`` script to match.

To uninstall, run:

    conda remove --name rlgpu --all

For troubleshooting check docs `docs/index.html`
<br></br>

## Usage

```bash
conda activate rlgpu
cd Isaacgym-drlgrasp/rlgpu
python train --task BaxterCabinet
```

You can choose to use PPO or SAC in `train.py` 
## TODO

**Still in development**
<br></br>

## Contributing

See [the contributing file](CONTRIBUTING.md)!
<br></br>

## License

[MIT © Richard McRichface.](../LICENSE)

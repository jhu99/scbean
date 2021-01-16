# Installation

- **Create conda enviroment**

For more information about conda environment, see this [tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

```shell
$ conda create -n scbean python=3.6
$ conda activate scbean
```
-  Install VIPCCA from pypi

```shell
$ pip install scbean
```

- **Install VIPCCA from GitHub source code**

```shell
$ git clone https://github.com/jhu99/scbean.git
$ cd ./scbean/
$ pip install .
```

**Note**: Please make sure that the `pip` is for python>=3.6. The current release depends on tensorflow with version 2.4.0. Install tenserfolow-gpu if gpu is avialable on the machine.


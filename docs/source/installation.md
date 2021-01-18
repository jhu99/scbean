# Installation

- **Create conda environment**

For more information about conda environment, see this [tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

```shell
$ conda create -n scbean python=3.8
$ conda activate scbean
```
-  Install scbean from pypi

```shell
$ pip install scbean
```

- **Install scbean from GitHub source code**

```shell
$ git clone https://github.com/jhu99/scbean.git
$ cd ./scbean/
$ pip install .
```

**Note**: 

- Please make sure your python version >= 3.7. The current release depends on tensorflow with version 2.4.0. Install tenserfolow-gpu if gpu is avialable on the machine.

- If there is a need to run large data sets, we provide version 1.1.1 (depending on tensorflow 1.15.1), which uses sparseTensor to reduce memory usage.

	```shell
	$ pip install scbean==1.1.1
	```




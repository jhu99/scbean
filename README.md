# VIPCCA
Variational inference of probabilistic canonical correlation analysis


# Creating an environment from an environment.yml file
cd $workspace

conda env create -f enviroment_cuda.yml or enviroment_cpu.yml

conda activate scEnv

conda env list

# install VIPCCA
tar -zxvf scxx.x.x.x.tar.gz

pip install -e ./scxx/

# run VIPCCA following example code

# reference
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


import setuptools
from setuptools import setup
from setuptools import Extension, dist, find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
      'scanpy',
      'tensorflow',
      'anndata',
      'scipy',
      'pandas',
      'seaborn',
      'keras',
      'python-igraph',
      'louvain',
      'h5py'
]
setup(name='scbean',
      version='0.3.0',
      description='integration',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/JHuLab/VIPCCA',
      author='Jialu Hu',
      author_email='jialuhu@umich.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False,
      python_requires='>=3.6',)

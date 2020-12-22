import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='VIPCCA',
      version='0.1.5',
      description='VIPCCA',
      long_description=long_description,
      url='https://github.com/JHuLab/VIPCCA',
      author='Jialu Hu',
      author_email='jialuhu@umich.edu',
      license='MIT',
      packages=['VIPCCA'],
      zip_safe=False,
      python_requires='>=3.6',)

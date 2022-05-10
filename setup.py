# Created by Elias Hanna 
# Date: 22/11/21

from setuptools import setup, find_packages

setup(name='model_init_study',
      install_requires=['gym', 'numpy'],
      version='1.0.0',
      packages=find_packages(),
      #include_package_data=True,
      author="Elias Hanna",
      author_email="h.elias@hotmail.fr",
      description="Study on the impact of various initialization methods on the initial capabilities of an ensemble of probabilistics neural networks",
)

from setuptools import setup
from pytorch_training import __version__

setup(
    name='pytorch_training',
    version=__version__,
    packages=['pytorch_training'],
    url='https://github.com/DKuzn/pytorch-training',
    license='LGPLv3',
    author='Dmitry Kuznetsov',
    author_email='DKuznetsov2000@outlook.com',
    description='Utilities to train PyTorch models',
    install_requires=['torch', 'tqdm']
)
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    description = f.read()

setup(
    name='stochdepth',
    version='0.1.0',
    description='A simple hook based implementation of "Deep Networks with Stochastic Depth" for torchvision resnets.',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Thomas Pönitz',
    author_email='tasptz@gmail.com',
    url='https://github.com/tasptz/pytorch-stochastic-depth',
    packages=['stochdepth'],
    install_requires=requirements,
    license='MIT',
    platforms=['any']
)

from setuptools import setup

with open('README.md') as f:
    description = f.read()

setup(
    name='stochdepth',
    version='0.1.2',
    description='A simple hook based implementation of "Deep Networks with Stochastic Depth" for torchvision resnets.',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Thomas Pönitz',
    author_email='tasptz@gmail.com',
    url='https://github.com/tasptz/pytorch-stochastic-depth',
    packages=['stochdepth'],
    license='MIT',
    platforms=['any'],
    install_requires=['torch>=1.8', 'torchvision>=0.9']
)

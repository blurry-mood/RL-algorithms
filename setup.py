from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

VERSION = 'v0.0.2'

# Read the requirements from the TXT file
with open(path.join(here, 'requirements.txt')) as f:
    requirements = [req for req in f.read().split('\n') if not ('#' in req or req == '')]
        
setup(
    name='rl_algorithms',
    version=VERSION,
    author='Ayoub Assis',
    author_email='assis.ayoub@gmail.com',
    url='https://github.com/blurry-mood/RL-algorithms',
    license='LICENSE',
    packages=find_packages(exclude=('simulations', 'utils')),
    keywords='pytorch reinforcement learning q-learning deep-q-learning',
    description='Implementation of reinforcement learning algorithms',
    long_description_content_type='text/markdown',

    install_requires=requirements,
    python_requires='>=3.7',

)
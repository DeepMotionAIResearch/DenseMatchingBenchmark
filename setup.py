from setuptools import find_packages
from setuptools import setup

setup(
    name="dmb",
    version="1.0",
    author="Youmi, Minwell",
    description="dense matching benchmark in pytorch",
    packages=find_packages(exclude=("tests")),
)


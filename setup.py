"""pip install . -- installs the pure-Python PyVega client."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
)

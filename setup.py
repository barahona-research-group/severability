"""Setup."""
from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.0"

setup(
    name="Severability",
    version=__version__,
    author="Yun William Yu",
    install_requires=[
        "numpy>=1.18.1",
    ],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
)

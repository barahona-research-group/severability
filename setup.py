"""Setup."""
from setuptools import find_packages
from setuptools import setup

__version__ = "0.2.0"

setup(
    name="Severability",
    version=__version__,
    author="Yun William Yu, Juni Schindler, Cameron Taylor",
    author_email="juni.schindler19@imperial.ac.uk",
    install_requires=[
        "matplotlib",
        "numpy>=1.18.1",
        "scipy",
        "tqdm",
    ],
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
)

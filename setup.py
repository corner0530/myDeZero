from setuptools import setup

setup(
    name="myDeZero",
    version="0.1.0",
    description="",
    author="",
    packages=["myDeZero"],
    install_requires=["numpy", "matplotlib"],
    package_dir={"myDeZero": ["steps"]},
)

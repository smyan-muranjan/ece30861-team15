from setuptools import find_packages, setup

setup(
    name="ece30861-team15",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gitpython",
        "requests",
        "huggingface-hub",
    ],
)

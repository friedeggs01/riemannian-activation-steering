from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="my_llm_utils",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,  # Include requirements from requirements.txt
)
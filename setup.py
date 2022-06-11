from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(),
    version="0.2.0",
    description="LivingNER project ",
    author="maranedah",
    license="MIT",
    install_requires=install_requires,
    include_package_data=True,
)

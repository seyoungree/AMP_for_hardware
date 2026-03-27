from setuptools import find_packages, setup


setup(
    name="rsl_rl",
    version="1.1.0",
    author="Nikita Rudin",
    author_email="rudinn@ethz.ch",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Fast and simple RL algorithms implemented in PyTorch",
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.5.0",
        "torchvision>=0.20.0",
    ],
)

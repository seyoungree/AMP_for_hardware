from setuptools import find_packages, setup


setup(
    name="legged_gym",
    version="1.1.0",
    author="Nikita Rudin",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="rudinn@ethz.ch",
    description="Legged robot AMP training utilities and legacy Isaac Gym environments",
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy",
        "rsl-rl",
        "torch>=2.5.0",
    ],
    extras_require={
        "legacy_isaacgym": ["isaacgym"],
    },
)

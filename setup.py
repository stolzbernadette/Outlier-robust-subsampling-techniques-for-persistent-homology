from setuptools import setup, find_packages

setup(
    name='ph_landmarks',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'ripser>=0.6.0',
        'ray>=2.0.0',
        'pytest>=6.2.0',
    ],
    python_requires='>=3.7',
)

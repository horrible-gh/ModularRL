from setuptools import setup, find_packages

setup(
    name='modular_rl',
    version='0.4.3',
    description='ModularRL package',
    author='sjm',
    author_email='shinjpn1@gmail.com',
    url='https://github.com/horrible-gh/ModularRL',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.22,<1.24',
        'torch>=2.0.1',
        'gym>=0.26.2',
        'LogAssist>=1.0.7',
        'tensorflow>=2.13.0',
        'tensorflow_probability>=0.20.1',
    ],
)

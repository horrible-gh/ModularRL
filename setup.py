from setuptools import setup, find_packages

setup(
    name='modular_rl',
    version='0.1.0.dev',
    description='ModularRL package',
    author='sjm',
    author_email='shinjpn1@gmail.com',
    url='https://shinjpn-gitea.duckdns.org/gitea/sjm/ModularRL.git',
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
        'numpy>=1.18.0',
        'torch>=1.24.2',
        'gym>=0.23.0',
    ],
)

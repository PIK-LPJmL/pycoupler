from setuptools import setup

# load the README file and use it as the long_description
with open('README.md', 'r') as f:
    readme = f.read()

# load the requirements file and use it as install_requires
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pycoupler',
    description='Python implementation of a package for operating and ' +
                'coupling LPJmL.',
    long_description=readme,
    long_description_content_type='text/markdown',
    version='0.3.0',
    author='Jannes Breier',
    author_email='jannes.breier@pik-potsdam.de',
    url='https://gitlab.pik-potsdam.de/lpjml/pycoupler',
    packages=['pycoupler'],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=required
)

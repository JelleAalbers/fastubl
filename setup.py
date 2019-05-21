import setuptools

readme = open('README.md').read()
requirements = open('requirements.txt').read().splitlines()

setuptools.setup(name='fastubl',
                 author='Jelle Aalbers',
                 version='0.0.1',
                 url='https://github.com/JelleAalbers/fastubl',
                 description='Fast unbinned likelihood Toy MCs',
                 package_dir={'fastubl': 'fastubl'},
                 packages=setuptools.find_packages(),
                 long_description=readme,
                 long_description_content_type='text/markdown',
                 install_requires=requirements,
                 setup_requires=['pytest-runner'],
                 tests_require=requirements + ['pytest'],
                 zip_safe=False)

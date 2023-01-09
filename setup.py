from setuptools import setup

setup(
    name='COSMO_TL',
    version='0.1',    
    description='COSMO-TL: A Python package for the Transfer Learning of COSMO-SAC data from the Virginia Tech Database to the NIST THERMOML Database',
    author='Eric C. Fonseca',
    author_email='ericfonseca@ufl.edu',
    license='BSD 2-clause',
    packages=['COSMO_TL'],
    install_requires=['dask','matplotlib', 'seaborn', 'scipy', 'numpy',],
    package_data={'COSMO_TL': ['data/data*']},

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

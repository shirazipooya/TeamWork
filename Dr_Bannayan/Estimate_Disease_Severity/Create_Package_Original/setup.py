from setuptools import setup, find_packages
 
classifiers = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Education',
    'Operating System :: POSIX :: Linux',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]
 
setup(
    name='EDS',
    version='0.5',
    description='EDS FOR ORIGINAL CODE',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='',  
    author='M. B.',
    author_email='mb@yahoo.com',
    license='MIT', 
    classifiers=classifiers,
    keywords='Disease', 
    packages=find_packages(),
    install_requires=['numpy', 'pandas'] 
)
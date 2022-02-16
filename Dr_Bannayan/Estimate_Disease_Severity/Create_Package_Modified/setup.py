from setuptools import setup, find_packages
 
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]
 
setup(
    name='TestMB',
    version='0.7',
    description='Test',
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
from distutils.core import setup

setup(
    name='ThreadPy',
    author='Eelco Hoogendoorn',
    author_email='hoogendoorn.eelco@gmail.com',
    packages=['threadpy'],
    scripts=[],
    url='http://pypi.python.org/pypi/ThreadPy/',
    license='LICENSE.txt',
    description='nd-array aware GPU kernels',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy",
    ],
)

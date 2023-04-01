
import setuptools


def get_version():
    with open('vismo/version.py') as f:
        exec(f.read())
    return locals()['__version__']


def get_readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='vismo',
    version=get_version(),
    license='MIT',
    author='moreih29',
    author_email='moreih29@gmail.com',
    description='Pytorch Vision Models',
    long_description=get_readme(),
    packages=setuptools.find_packages(),
    url='https://github.com/moreih29/vismo',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch vision model classification object detection segmentation pose estimation',
    include_package_data=True,
    install_requires=[],
    python_requires='>=3.7',
)
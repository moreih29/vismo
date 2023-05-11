
from setuptools import setup, find_packages

from vismo import __version__


def get_readme():
    with open('README.md') as f:
        return f.read()


def get_requires():
    requires = [
        'opencv-python>=4.5',
        'scikit-learn>=1.1.0',
        'tqdm>=4.63',
    ]
    return requires


def main():
    setup(
        name='vismo',
        version=__version__,
        license='MIT',
        author='moreih29',
        author_email='moreih29@gmail.com',
        description='Pytorch Vision Models',
        long_description=get_readme(),
        packages=find_packages(),
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
        install_requires=get_requires(),
        python_requires='>=3.7',
    )
    

if __name__ == '__main__':
    main()
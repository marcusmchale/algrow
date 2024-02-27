from setuptools import setup

setup(
    name='algrow',
    version='0.6.2',
    packages=['src'],
    scripts=['algrow.py'],
    url='https://github.com/marcusmchale/algrow',
    license='LICENSE.txt',
    author='Marcus McHale',
    author_email='marcus.mchale@universityofgalway.ie',
    description="Image segmentation using alpha hulls with additional annotation and growth analysis",
    install_requires=[
        "configargparse~=1.7",
        "numpy~=1.26.2",
        "pandas~=2.1.3",
        "scipy~=1.11.1",
        "scikit-image~=0.22.0",
        "alphashape~=1.3.1",
        "matplotlib~=3.8.2",
        "open3d~=0.17.0"
    ]
)

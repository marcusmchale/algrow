from setuptools import setup

setup(
    name='algrow',
    version='0.4',
    packages=['src'],
    scripts=['algrow.py'],
    url='https://github.com/marcusmchale/algrow',
    license='LICENSE.txt',
    author='Marcus McHale',
    author_email='marcus.mchale@universityofgalway.ie',
    description='Alpha shape defined colour space for automated growth rate determination in multiplexed images of macroalgal discs"',
    install_requires=[
        "configargparse~=1.5.5",
        "numpy~=1.25.0",
        "pandas~=2.0.3",
        "scipy~=1.11.1",
        "scikit-image~=0.21.0",
        "alphashape~=1.3.1",
        "matplotlib~=3.7.1",
        "open3d~=0.17.0"
    ]
)

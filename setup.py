from setuptools import setup

setup(
    name='discgrow',
    version='0.2',
    packages=['src'],
    scripts=['discgrow.py'],
    url='https://github.com/marcusmchale/discgrow',
    license='LICENSE.txt',
    author='marcus',
    author_email='marcus.mchale@nuigalway.ie',
    description='Graph-based detection of target superpixels for automated growth rate determination in multiplexed images of macroalgal discs"',
    install_requires=["numpy", "opencv-python", "scipy", "scikit-image", "matplotlib", "pandas", "setuptools"]
)

from setuptools import setup

setup(
    name='src',
    version='0.1.0',
    packages=['src'],
    scripts=['run.py'],
    url='https://github.com/marcusmchale/discgrow',
    license='LICENSE.txt',
    author='marcus',
    author_email='marcus.mchale@nuigalway.ie',
    description='Automated multiplexed area quantification for growth rate determination of macroalgal leaf discs',
    install_requires=["numpy", "opencv-python", "scipy", "scikit-image", "matplotlib"]
)
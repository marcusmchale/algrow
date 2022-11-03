from setuptools import setup

setup(
    name='discgrow',
    version='0.1.4',
    packages=['src'],
    scripts=['discgrow.py'],
    url='https://github.com/marcusmchale/discgrow',
    license='LICENSE.txt',
    author='marcus',
    author_email='marcus.mchale@nuigalway.ie',
    description='Automated multiplexed area quantification for growth rate determination of macroalgal leaf discs',
    install_requires=["numpy", "opencv-python", "scipy", "scikit-image", "matplotlib", "pandas", "setuptools"]
)

from setuptools import setup

setup(
    name='algrow',
    version='0.6.3-5',
    packages=['algrow'],
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"resources": ["resources/*.png", "resources/*.txt"]},
    entry_points={'console_scripts': ['algrow=algrow.launch:run']},
    url='https://github.com/marcusmchale/algrow',
    license='LICENSE.txt',
    author='Marcus McHale',
    author_email='marcus.mchale@universityofgalway.ie',
    description="Image segmentation using alpha hulls with additional annotation and growth analysis",
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
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

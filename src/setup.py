import os.path as pt
from setuptools import setup, find_packages

PACKAGE_DIR = pt.abspath(pt.join(pt.dirname(__file__)))

packages = find_packages(PACKAGE_DIR)

package_data = {
    package: [
        '*.py',
        '*.txt',
        '*.json',
        '*.npy'
    ]
    for package in packages
}

with open(pt.join(PACKAGE_DIR, 'requirements.txt')) as f:
    dependencies = [l.strip(' \n') for l in f]
    # Pillow-simd==9.0.0.post1 ?

setup(
    name='eoe',
    version='0.1',
    classifiers=[
        'Development Status :: 4 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.10'
    ],
    keywords='deep-learning anomaly-detection one-class-classification outlier-exposure',
    packages=packages,
    package_data=package_data,
    install_requires=dependencies,
)

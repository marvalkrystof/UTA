from pathlib import Path

from setuptools import setup, find_packages


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name='uta_solver',
    version='1.0',
    description='UTA (Utilités Additives) - Multi-criteria decision analysis library',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT',
    license_files=('LICENSE',),
    author='Kryštof Marval',
    author_email='marval.krystof@seznam.cz',
    packages=find_packages(
        exclude=[
            'tests',
            'tests.*',
            'uta_solver.tests',
            'uta_solver.tests.*',
            'frontend',
            'frontend.*',
        ]
    ),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'frontend': [
            'streamlit>=1.20.0',
            'plotly>=5.0.0',
            'streamlit-sortables>=0.3.1',
        ],
        'dev': [
            'pytest>=7.0.0',
        ],
    },
    python_requires='>=3.11',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)

from setuptools import setup, find_packages

setup(
    name='bikeshare_prediction',
    version='0.1.0',
    description='A package for predicting bike share usage',
    author='Your Name',
    author_email='your_email@example.com',
    url='https://github.com/yourusername/bikeshare_prediction',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
        'pytest',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

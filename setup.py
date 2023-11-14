from setuptools import setup, find_packages
setup(
    name = 'sklearn-sid',
    packages = [
        'sklearn_sid',
        'sklearn_sid.base',
        'sklearn_sid.regressors',
        'sklearn_sid.utils',
                ],
    install_requires=[
        "scikit-learn==1.2.2",
        "numpy==1.24.2",
        #"pandas==2.0.0",
    ]
)
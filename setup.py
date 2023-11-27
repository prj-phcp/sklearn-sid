from setuptools import setup, find_packages
setup(
    name = 'sklearn-sid',
    packages = [
        'sklearn_sid',
        'sklearn_sid.base',
        'sklearn_sid.preprocessors',
        'sklearn_sid.regressors',
        'sklearn_sid.utils',
                ],
    install_requires=[
        "scikit-learn==1.2.2",
        "numpy==1.26.2",
        "scipy==1.11.4",
        "statsmodels==0.14.0"
    ]
)
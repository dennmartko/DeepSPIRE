from setuptools import setup, find_packages

setup(
    name="DeepSPIRE",
    version="0.1.0",
    description="A code for super-resolving Herschel SPIRE images using SwinUnet Transformer.",
    package_dir={'deepSPIRE': 'scripts'},
    packages=['deepSPIRE'] + ['deepSPIRE.' + pkg for pkg in find_packages(where='scripts')],
)

from setuptools import setup, find_packages

setup(
    name="config_iq",
    version="1.0.0",
    description="Hardware Sizing Tool for LLM Inference",
    author="Varcas",
    packages=find_packages(),
    package_data={
        "config_iq": ["data/*.json"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "config_iq=config_iq.cli.main:main",
        ],
    },
)

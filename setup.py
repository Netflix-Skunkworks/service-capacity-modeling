import setuptools

setuptools.setup(
    name="service-capacity-modeling",
    author="Joseph Lynch",
    author_email="josephl@netflix.com",
    versioning="distance",
    setup_requires="setupmeta",
    description="Contains utilities for modeling database capacity on a cloud",
    packages=setuptools.find_packages(exclude=("tests*", "notebooks*")),
    install_requires=[
        "pydantic",
        "scipy",
        "numpy",
        'importlib_resources; python_version < "3.7"',
    ],
    extras_require={
        "aws": ["boto3"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": []},
    include_package_data=True,
    package_data={
        "": [
            "hardware/profiles/profiles.txt",
            "hardware/profiles/shapes/*.json",
            "hardware/profiles/pricing/**/*.json",
        ]
    },
)

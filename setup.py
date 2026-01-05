import setuptools

setuptools.setup(
    name="service-capacity-modeling",
    author="Joseph Lynch",
    author_email="josephl@netflix.com",
    versioning="distance",
    setup_requires=["setupmeta"],
    description="Contains utilities for modeling capacity for pluggable workloads",
    python_requires=">=3.10,<3.13",
    packages=setuptools.find_packages(exclude=("tests*", "notebooks*")),
    install_requires=[
        "pydantic>2.0",
        "scipy",
        "numpy",
        "isodate",
    ],
    # https://github.com/python/typeshed/issues/15220
    extras_require={  # type: ignore[arg-type]
        "aws": ["boto3"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "auto-shape = service_capacity_modeling.tools.auto_shape:main",
            "fetch-pricing = service_capacity_modeling.tools.fetch_pricing:main",
        ]
    },
    include_package_data=True,
    # https://github.com/python/typeshed/issues/15220
    package_data={  # type: ignore[arg-type]
        "": [
            "hardware/profiles/profiles.txt",
            "hardware/profiles/shapes/**/*.json",
            "hardware/profiles/pricing/**/*.json",
        ]
    },
)

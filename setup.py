from setuptools import setup, find_packages


TESTS_REQUIRES = [
    "black>=20.8b1",
    "mypy",
    "pylint",
    "types-pyyaml",
    "types-requests",
]

setup(
    name="subsurface-movie-renderer",
    description="Render 3D movies for subsurface data",
    long_description_content_type="text/markdown",
    url="https://github.com/equinor/subsurface-movie-renderer",
    author="R&T Equinor",
    packages=find_packages(exclude=["tests"]),
    entry_points={
        "console_scripts": [
            "subsurface_movie_renderer=subsurface_movie_renderer.command_line:main"
        ],
    },
    install_requires=[
        "numpy",
        "pyyaml>=5.1",
        "requests>=2.20",
        "scipy",
        "tqdm>=4.8",
    ],
    setup_requires=["setuptools_scm~=3.2"],
    extras_require={
        "tests": TESTS_REQUIRES,
    },
    python_requires="~=3.8",
    use_scm_version=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

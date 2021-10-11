[![Build Status](https://github.com/equinor/subsurface-movie-renderer/workflows/subsurface-movie-renderer/badge.svg)](https://github.com/equinor/subsurface-movie-renderer/actions?query=branch%3Amain)
[![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8%20|%203.9-blue.svg)](https://www.python.org/)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Subsurface movie renderer

### Installation

You can install the code by cloning the repository, changing to the cloned directory and run
```bash
pip install .
```

Note that you will also need two non-Python dependencies:
- `blender` version 2.79 or below (https://www.blender.org/)
- `ffmpeg` (https://www.ffmpeg.org/)

### Usage

In order to test successfull installation, you can try running the example setup:
```
subsurface_movie_renderer example_config/user_configuration.yml
```
In order to see all command line options, you can run:
```
subsurface_movie_renderer --help
```

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Added a `RecipeGenerationModel` class
- Added a [recipe generation model](models/dugan2020/Readme.md) from [Dugan et al. (2020)](https://arxiv.org/abs/2010.03070)
- Added a `TruecasingModel` class
- Added an RNN-based truecaser from [Susanto et al. (2016)](https://aclanthology.org/D16-1225/) based on an implementation [here](https://github.com/mayhewsw/pytorch-truecaser).

## [v0.0.1](https://github.com/danieldeutsch/repro/repro/tag/0.0.1) - 2021-07-22
### Added
- Initial prototype of the library with `setup` and `predict` commands as well as implementations of [Gupta et al. (2020)](models/gupta2020/Readme.md), [Lewis et al. (2020)](models/lewis2020/Readme.md), and [Liu & Lapata (2019)](models/liu2019/Readme.md).

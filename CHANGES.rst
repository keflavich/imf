Changelog
=========

2026.07.11
----------

This release accompanies the paper "The IMF package: a toolkit implementing
mass functions and statistical tools to analyze them" (Richardson, Ginsburg &
Koposov).

New features
^^^^^^^^^^^^
- Added a dedicated ``sampling`` module and substantially expanded the sampling
  operations for mass functions.
- Added star-cluster creation with independent preservation of stellar
  multiples during sampling.
- Added a multiplicity function and mass-ratio / mass-conversion utilities.
- Added a basic Integrated Galactic IMF (IGIMF) utility.

Bug fixes
^^^^^^^^^
- Fixed broken behavior in the broken-power-law ``integrate`` routine (#56).
- Accounted for the CDF behavior of Schechter functions.
- Fixed an incorrect ``super()`` call.
- Improved performance of the multiplicity calculation.
- Updated protostellar/PN core-mass-function core handling and the core setter
  method.

Documentation
^^^^^^^^^^^^^
- Added a new documentation page describing the sampling functionality.
- Documented the new sampling functions and updated examples and docs to use
  the new sampling API.

Maintenance
^^^^^^^^^^^
- Retained ``make_cluster`` as a deprecated compatibility function.
- Consolidated the license to BSD 3-Clause across all files and packaging
  metadata; added ``.zenodo.json`` and ``CITATION.cff`` with full author,
  affiliation, and ORCID metadata.
- Pre-commit, autopep8, and notebook-output cleanups.

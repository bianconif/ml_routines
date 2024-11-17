# A tiny collection of Python routines for machine learning

## Main features

### Performance estimation of feature sets for supervised classification
- `performance_estimation/internal_validation`
  - Internal validation given one feature set
    - Stratified k-fold
    - Stratified shuffle split
- `performance_estimation/cross_validation`
  - Cross validation given two feature sets

### Combination of classifiers
- Combination via early fusion (feature concatenation)
- Combination via late fusion
  - Hard-level combination: majority voting
  - Soft-level combination: product, sum and max rule

## Dependencies
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## License
The code in this repository is distributed under [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

## Disclaimer
The information and content available on this repository are provided with no warranty whatsoever. Any use for scientific or any other purpose is conducted at your own risk and under your own responsibility. The authors are not liable for any damages - including any consequential damages - of any kind that may result from the use of the materials or information available on this repository or of any of the products or services hereon described.

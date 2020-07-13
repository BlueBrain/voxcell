Changelog
=========

Version 2.7.1
-------------

- Changed processing of properties of CellCollection that are pandas.Categorical. A special rule for
  string properties is applied. If unique values of a property make less than half of its all values
  then it is loaded as pandas.Categorical.

Version 2.7.0
-------------

- Introduce serialization of CellCollection to SONATA format. It is the preferred choice. MVD3 can
  be saved/loaded only when the direct file extension `.mvd3` is used.

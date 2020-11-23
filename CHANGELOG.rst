Changelog
=========

Version 2.7.4
--------------

Improvements
~~~~~~~~~~~~
- Remove the NodePopulation implementation and hard deprecate the class. Deprecate the
  node_population.py module itself.


Version 2.7.3
--------------

New Features
~~~~~~~~~

- Add a function to retrieve the region ids of layers in a brain region



Version 2.7.2
--------------

Bug Fixes
~~~~~~~~~

- Force the version of h5py to h5py<3.0.0. h5py==3.0.0 dropped the python2 support and changed the
  string behaviors.


Version 2.7.1
-------------

- Set the nrrd header_ field `kinds` with the value `['vector', 'domain', 'domain', domain']` when saving an
  nrrd file which holds a non-scalar vector field over a volume. This change enables visualization of
  direction vectors (3D vector field) and orientations (quaternions, 4D) with ITK-SNAP_ and 3D Slicer_.

- Changed processing of properties of CellCollection that are pandas.Categorical. A special rule for
  string properties is applied. If unique values of a property make less than half of its all values
  then it is loaded as pandas.Categorical.

- Deprecated the Hierarchy class in profit of the RegionMap. The Hierarchy class should be removed
  in 2.8.0. Redo the docs for the RegionMap object.

- Changed saving of `CellCollection`. Raise an error if there is a `None` or `np.NaN` in
  `CellCollection`.

- Fixed the orientation loading for sonata files in `CellCollection`. Two different formats
  exist : the euler's angles and the quaternions.

    - use quaternions if all "orientation_[x|y|z]" are present
    - if some of the "orientation_[x|y|z]" fields are here but not all. Raise.
    - if orientations and rotation_angles are present use quaternions
    - if no quaternions and some of the rotation_angles use the eulers angles
      and assign 0 to the missing ones.

- Added a orientation_format property to the `CellCollection` class. This allows the user to choose
  which sonata orientation format she/he wants to use.

- Added a size function to `CellCollection`.

- Check the sizes of the orientations/positions/properties before saving.

Version 2.7.0
-------------

- Introduce serialization of CellCollection to SONATA format. It is the preferred choice. MVD3 can
  be saved/loaded only when the direct file extension `.mvd3` is used.


.. _header: http://teem.sourceforge.net/nrrd/format.html#kinds
.. _ITK-SNAP: http://www.itksnap.org/pmwiki/pmwiki.php
.. _Slicer: https://www.slicer.org/
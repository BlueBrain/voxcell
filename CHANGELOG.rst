Changelog
=========

Version 3.1.10
--------------

- Allow to specify ``index_offset`` (0 or 1) in ``CellCollection.as_dataframe()`` and ``CellCollection.from_dataframe()`` (#34)
- Fix nrrd header format (#38)

Version 3.1.9
-------------

- Add the VoxelData.value_to_indices() method (#33)

Version 3.1.8
-------------

- Add unravel method to ValueToIndexVoxels (#31)
- Add RegionMap.from_dataframe() and RegionMap.as_dict() methods (#32)

Version 3.1.7
-------------

- Add ValueToIndexVoxels to simplify using indices instead of masks (#30)

Version 3.1.6
-------------

- Update read the docs (#25)
- Fix warnings (#27)

Version 3.1.5
-------------

- Fix outer value lookup with vector data (#19)
- Fix compatibility with groupby from pandas 2.0

Version 3.1.4
-------------
- switch hemisphere constants to be in line with Allen Institute

Version 3.1.3
-------------
- Add a RegionMap conversion to Pandas DataFrame (#6)
- Add a function to compute the spatial intersection of a line-segment with a VoxelData object (#11)

Version 3.1.1
-------------
- Add documentation for ROI Mask, Orientation and scalar file formats
- CellCollection::save_sonata allows for a `mode`; output file can be appended to, or overwritten.
- CellCollection can now read multiple SONATA popluations.

Version 3.1.0
-------------
- Add str/repr to CellCollection.
- CellCollection.save_sonata learned how to `force_library` so that the user
  can control the behavior of `@library` creation.
- Use `Rotation.from_euler` from scipy in `angles_to_matrices` (fix rotation around `x` and `z` axis).
- Add `values_to_region_attribute` and `values_to_hemisphere` to convert values returned by `VoxelData.lookup()`.

Version 3.0.2
-------------
- Set the dtype of the default zero offset of VoxelData to np.float32 (consistent with the other assignments)
- Use pytest for tests instead of nosetests
- Cleanup deprecation warnings emitted by numpy/pandas

Version 3.0.1
-------------

Improvements
~~~~~~~~~~~~
- Use sphinx-bluebrain-theme for documentation
- Add missing API to documentation

Version 3.0.0
-------------


Python 2 compatibility
~~~~~~~~~~~~~~~~~~~~~~

- Dropped Python 2 support


Bug Fixes
~~~~~~~~~

- Fixed issue with H5PY >= 3.0.0 string type behavior. Changes in H5py >= 3.0.0 API affected the
  reading of h5 files by returning string fields as bytes fields. The recommended use of the method
  `Dataset.asstr()` to return string field is implemented in this version.

Removed modules
~~~~~~~~~~~~~~~

- Removed deprecated module `voxcell.positions`

- Removed deprecated module `voxcell.build`

- Removed deprecated module `voxcell.core`

- Removed deprecated module `voxcell.voxell_data`

- Removed deprecated function `voxcell.vector_field.generate_homogeneous_field`:
- Removed deprecated function `voxcell.vector_field.calculate_fields_by_distance_from`:
- Removed deprecated function `voxcell.vector_field.calculate_fields_by_distance_to`:
- Removed deprecated function `voxcell.vector_field.calculate_fields_by_distance_between`:
- Removed deprecated function `voxcell.vector_field.compute_cylindrical_tangent_vectors`:
- Removed deprecated function `voxcell.vector_field.compute_hemispheric_spherical_tangent_fields`:



Version 2.7.4
--------------

Improvements
~~~~~~~~~~~~
- Remove the NodePopulation implementation and hard deprecate the class. Deprecate the
  node_population.py module itself.


Version 2.7.3
--------------

New Features
~~~~~~~~~~~~

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

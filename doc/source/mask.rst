Mask Image for Region of Interest (ROI)
=======================================

.. _mask_roi:

For validating the circuits we would need to specify "regions of interest" for a given atlas (for example, pick a sphere within a layer to measure cell density). A mask image is generated for given specific shape and stored in NRRD format with the following metadata.

.. bbp_table_section:: Specification
    :description:
     The meta data stored in the NRRD file should follow the specification below:

    .. bbp_table_value:: dimension
        :type: integer
        :value: 3
        :comment: dimension of the image is always 3 for volumetric data

    .. bbp_table_value:: encoding
        :type: string
        :value: gzip
        :comment: use 'gzip' for compressing the data

    .. bbp_table_value:: endian
        :type: string
        :value: little
        :comment: use 'little' for x86 machines

    .. bbp_table_value:: kinds
        :type: list of string
        :value: ['domain', 'domain', 'domain']
        :comment: use 'domain' to indicate that the axis is image axis

    .. bbp_table_value:: sizes
        :type: list of integer
        :value: [{size_x}, {size_y}, {size_z}]
        :comment: the size of the volume in x, y, z dimension

    .. bbp_table_value:: space directions
        :type: 3d matrix
        :value:
        :comment: A 3D matrix indicating the orientation of the image data, with the value indicating the spacing of the voxel

    .. bbp_table_value:: space origin
        :type: list of float
        :value: [{origin_x}, {origin_y}, {origin_z}]
        :comment: physical coordinates of the image origin

    .. bbp_table_value:: space
        :type: string
        :value: {x_orientation}-{y_orientation}-{z_orientation}
        :comment: the orientation of the image data, should be consistent with the space direction

    .. bbp_table_value:: type
        :type: string
        :value: uchar
        :comment: the type of the values stored in the voxels is unsigned char

Example:

::

  {
    u'space origin': ['-46.540000915527344', '-152.15999984741211', '-152'],
    u'space directions': [['16', '0', '0'], ['0', '16', '0'], ['0', '0', '16']],
    u'sizes': [308, 495, 464],
    u'space': 'left-posterior-superior',
    u'encoding': 'gzip',
    u'endian': 'little',
    u'kinds': ['domain', 'domain', 'domain'],
    u'type': 'unsigned char',
    u'dimension': 3
  }

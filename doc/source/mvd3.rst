MVD3
====

MVD3 is an iteration of previous file formats used at the Blue Brain Project.
It is meant to support 3 dimensional rotations of morphologies, and thus capable of modelling anatomically realistic circuits.
It is designed to be able to handle larger circuits and to preserve numbers without having to rely on string conversion routines.
The format is extensible as we foresee the appearance of new fields beyond the current classifications of cells.

Fields
------

The file represents a description of a group of cells. It contains, for each cell, the values of a set of properties or fields.

No field is mandatory and new ones may be added in the future. The container API provides a way to query the presence or absence of fields.

The following is a list of a few common fields:

- position: X, Y, Z, stored as 3 floats. The unit is micrometer.
- orientation: A quaternion, X Y Z W stored as 4 floats.
- etype: electrical type.
- mtype: morphological type.
- synapse_class: inhibitory/excitatory.

Container
---------

MVD3 uses HDF5 as the structured data container. It is the HPC standard for saving data, and there are libraries for accessing it with C++ and Python.

Floating-point numeric properties (like position and orientation) are stored as individual datasets under /cells.

Text-based properties where most of the entires will be duplicated (like etype, mtype and synapse_class) are stored as two datasets:

- One containing the list of all unique used values under /library/*
- One containing an index into the first one for each cell under /cells/properties/*

The following is an example of the structure of an HDF5 containing all known cell properties: ::

    /                               Group
    /cells                          Group
    /cells/orientations             Dataset {N, 4}
    /cells/positions                Dataset {N, 3}
    /cells/properties               Group
    /cells/properties/etype         Dataset {N}
    /cells/properties/morphology    Dataset {N}
    /cells/properties/mtype         Dataset {N}
    /cells/properties/synapse_class Dataset {N}
    /library                        Group
    /library/etype                  Dataset {E}
    /library/morphology             Dataset {M}
    /library/mtype                  Dataset {T}
    /library/synapse_class          Dataset {S}

Where the cardinality is:

- N is the number of cells.
- M is the number of unique morphologies.
- E is the number of unique e-types.
- T is the number of unique m-types.
- S is the number of unique synapse classes.

On top of this, two attributes at top level "/" can be used to identify the file format. These are:

- Attribute: *version*. Value:  [3, 0]  (array with [mayor, minor]).
- Attribute: *format*. Value: "MVD" (string).

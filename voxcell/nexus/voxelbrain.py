"""Access to VoxelBrain.

https://nip.humanbrainproject.eu/documentation/user-manual.html#voxel-brain
"""

import abc
import json
import os
import urllib

import numpy as np
import requests

from voxcell import RegionMap, VoxelData, math_utils
from voxcell.exceptions import VoxcellError


def _download_file(url, filepath, overwrite, allow_empty=False):
    """Download file from `url` if it is missing."""
    if os.path.exists(filepath) and not overwrite:
        return filepath

    tmp_filepath = filepath + ".download"
    try:
        resp = requests.get(url, timeout=None)
        resp.raise_for_status()
        if not (allow_empty or resp.content):
            raise VoxcellError("Empty content")
        with open(tmp_filepath, "wb") as f:
            f.write(resp.content)
        os.rename(tmp_filepath, filepath)
    finally:
        try:
            os.unlink(tmp_filepath)
        except OSError:
            pass

    return filepath


class Atlas:
    """Helper class for atlas access."""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Init Atlas."""
        self._memcache = {}

    @staticmethod
    def open(url, cache_dir=None):
        """Get Atlas object to access atlas stored at URL."""
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme in ('', 'file'):
            return LocalAtlas(url)

        if parsed.scheme in ('http', 'https'):
            if not parsed.path.startswith('/api/analytics/atlas/releases/'):
                raise VoxcellError(f"Unexpected URL: '{url}'")

            if cache_dir is None:
                raise VoxcellError("`cache_dir` should be specified")

            return VoxelBrainAtlas(url, cache_dir)

        raise VoxcellError(f"Unexpected URL: '{url}'")

    @abc.abstractmethod
    def fetch_data(self, data_type):
        """Fetch `data_type` NRRD."""

    @abc.abstractmethod
    def fetch_hierarchy(self):
        """Fetch brain region hierarchy JSON."""

    def _check_cache(self, key, callback, memcache):
        if key in self._memcache:
            return self._memcache[key]
        result = callback()
        if memcache:
            self._memcache[key] = result
        return result

    def load_data(self, data_type, cls=VoxelData, memcache=False):
        """Load atlas data layer."""
        def _callback():
            return cls.load_nrrd(self.fetch_data(data_type))

        return self._check_cache(
            ('data', data_type, cls),
            callback=_callback,
            memcache=memcache
        )

    def load_region_map(self, memcache=False):
        """Load brain region hierarchy as RegionMap."""
        def _callback():
            return RegionMap.load_json(self.fetch_hierarchy())

        return self._check_cache(
            ('region_map',),
            callback=_callback,
            memcache=memcache
        )

    def get_region_mask(self, value, attr='acronym', with_descendants=True,
                        ignore_case=False, memcache=False):
        """Get VoxelData with 0/1 mask indicating regions matching `value`."""

        def _callback():
            rmap = self.load_region_map()
            brain_regions = self.load_data('brain_regions')
            region_ids = rmap.find(
                value, attr=attr, with_descendants=with_descendants,
                ignore_case=ignore_case
            )
            if not region_ids:
                raise VoxcellError(f"Region not found: '{value}'")
            result = math_utils.isin(brain_regions.raw, region_ids)
            return brain_regions.with_data(result)

        return self._check_cache(
            ('region_mask', value, attr, with_descendants),
            callback=_callback,
            memcache=memcache
        )


class VoxelBrainAtlas(Atlas):
    """Helper class for VoxelBrain atlas."""
    def __init__(self, url, cache_dir):
        """Init VoxelBrainAtlas."""
        super().__init__()
        self._url = url.rstrip("/")
        resp = requests.get(self._url, timeout=None)
        resp.raise_for_status()
        atlas_id = resp.json()[0]['id']
        assert self._url.endswith(atlas_id)
        self._cache_dir = os.path.join(cache_dir, atlas_id)
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def fetch_data(self, data_type):
        """Fetch `data_type` NRRD."""
        resp = requests.get(self._url + "/data", timeout=None)
        resp.raise_for_status()
        data_types = []
        for item in resp.json():
            if item['data_type'] == data_type:
                url = item['url']
                break
            data_types.append(item['data_type'])
        else:
            raise VoxcellError(
                # pylint: disable=consider-using-f-string
                "`data_type` should be one of ({0}), provided: {1}".format(
                    ",".join(data_types), data_type
                )
            )
        filepath = os.path.join(self._cache_dir, f"{data_type}.nrrd")
        return _download_file(url, filepath, overwrite=False)

    def fetch_hierarchy(self):
        """Fetch brain region hierarchy JSON."""
        url = self._url + "/filters/brain_region/65535"
        filepath = os.path.join(self._cache_dir, "hierarchy.json")
        return _download_file(url, filepath, overwrite=False)


class LocalAtlas(Atlas):
    """Helper class for locally stored atlas."""
    def __init__(self, dirpath):
        """Init LocalAtlas."""
        super().__init__()
        self.dirpath = dirpath

    def _get_filepath(self, filename):
        result = os.path.join(self.dirpath, filename)
        if not os.path.exists(result):
            raise VoxcellError(f"File not found: '{result}'")
        return result

    def fetch_data(self, data_type):
        """Return filepath to `data_type` NRRD."""
        return self._get_filepath(f"{data_type}.nrrd")

    def fetch_hierarchy(self):
        """Return filepath to brain region hierarchy JSON."""
        return self._get_filepath("hierarchy.json")

    def fetch_metadata(self):
        """Return filepath to metadata JSON."""
        return self._get_filepath("metadata.json")

    def load_metadata(self, memcache=False):
        """Load brain region metadata as dict."""

        def _callback():
            with open(self.fetch_metadata(), 'r', encoding='utf-8') as f:
                return json.load(f)

        return self._check_cache(('metadata',), callback=_callback, memcache=memcache)

    def get_layers(self):
        """Retrieve and cache the identifiers of each layer of a laminar brain region.

        For each layer of an annotated brain volume, the function returns the set of identifiers of
        all regions included in this layer.

        Note: this function relies on the existence of the files
            * hierarchy.json
            * metadata.json

        See get_layer for the description of the metadata.json file.

        Returns: tuple (names, ids) where `names` is a list of layer names where `ids` is a list
            of sets and. Each set contains the identifiers (ints) of the corresponding layer name
            (str).

        Raises: VoxcellError
            * if the hierarchy file or the metadata file doesn't exist.
            * if metadata.json doesn't contain the key "layers"
            * if the value of "layers" doesn't contain all the required keys:
                "names", "queries" and "attribute".
            * if the value  of "names" or "ids" is not a list, or if these objects are two lists
                of different lengths.
        """

        def _callback():
            metadata = self.load_metadata()

            if 'layers' not in metadata:
                raise VoxcellError('Missing "layers" key')
            layers = metadata['layers']

            if not all(
                key in metadata['layers'] for key in ['names', 'queries', 'attribute']
            ):
                err_msg = (
                    'Missing some "layers" key. The "layers" dictionary has '
                    'the following mandatory keys: "names", "queries" and "attribute"'
                )
                raise VoxcellError(err_msg)

            if not (
                isinstance(layers['names'], list)
                and isinstance(layers['queries'], list)
                and len(layers['names']) == len(layers['queries'])
            ):
                raise VoxcellError(
                    'The values of "names" and "queries" must be lists of the same length'
                )

            region_map = self.load_region_map()

            ids = [
                region_map.find(query, attr=layers['attribute'], with_descendants=True)
                for query in layers['queries']
            ]

            return (layers['names'], ids)

        return self._check_cache(
            ('get_layer_ids'),
            callback=_callback,
            memcache=True,
        )

    def get_layer(self, layer_index):
        """Retrieve the identifiers of a specified layer in laminar brain region.

        Given a layer of an annotated brain volume, the function returns the set of identifiers of
        all regions included in this layer.

        Note: this function relies on the existence of the files
            * hierarchy.json
            * metadata.json

        The content of metadata.json must be of the following form::

            {
                ...
                "layers": {
                    "names": [
                        "layer 1", "layer 2/3", "layer 4", "layer 5", "layer 6", "Olfactory areas"
                    ],
                    "queries": ["@.*1$", "@.*2/3$", "@.*4$", "@.*5$", ".*6[a|b]?$", "OLF"],
                    "attribute": "acronym"
                },
                ...
            }

        The strings of the `queries` list are in one-to-one correspondence with layer `names`.
        The layer `names` list is a user-defined string list which can vary depending on which
        hierarchy.json is used and which brain region is under scrutiny.
        Each query string is used to retrieve the region identifiers of the corresponding
        layer by means of the Atlas RegionMap object instantiated from the hierarchy.json file.
        The syntax of a query string is the same as the one in use for RegionMap.find:
        if the string starts with the symbol '@', the remainder is interpreted as a regular
        expression. Otherwise it is a plain string value and RegionMap.find looks for a
        character-wise full match.
        The value of `attribute` is `acronym` or `name`. This unique value applies for every query
        string.

        Returns: tuple (name, ids) where `name` is the name of the layer (str) with index
            `layer_index` and `ids` is the set of region identifiers of every region
            in this layer according to hierarchy.json.

        Raises: VoxcellError
            * if the hierarchy file or the metadata file doesn't exist.
            * if metadata.json doesn't contain the key "layers"
            * if the value of "layers" doesn't contain all the required keys:
                "names", "queries" and "attribute".
            * if the value  of "names" or "ids" is not a list, or if these objects are two lists
                of different lengths.
        """
        layer_names, layer_ids = self.get_layers()

        return (layer_names[layer_index], layer_ids[layer_index])

    def get_layer_volume(self, memcache=False):
        """Get VoxelData whose voxels are labeled with the layer indices of `brain_regions` voxels.

        Layer indices range from 1 to `number of layers` - 1. The layers of the atlas are defined
        within metadata.json as a list of RegionMap.find query strings.

        If two layer definitions in metadata.json involve the same region identifier,
        the voxels bearing this identifier will be labled with the largest layer index.

        Args:
            memcache: If True, use cache, Otherwise re-compute the layer volume.
                Defaults to False.

        Returns:
            VoxelData object with the same shape and same metadata as brain_regions. Voxels have
            uint8 labels (char type) and each voxel label corresponds to the voxel layer index.
        """

        def _callback():
            brain_regions = self.load_data('brain_regions')
            _, layer_ids = self.get_layers()
            layers = np.zeros_like(brain_regions.raw, dtype=np.uint8)
            for index, ids in enumerate(layer_ids, 1):
                mask = math_utils.isin(brain_regions.raw, ids)
                layers[mask] = index

            return brain_regions.with_data(layers)

        return self._check_cache(
            ('layer_volume'),
            callback=_callback,
            memcache=memcache,
        )

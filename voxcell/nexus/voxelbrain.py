"""
Access to VoxelBrain.

https://nip.humanbrainproject.eu/documentation/user-manual.html#voxel-brain
"""

import os
import abc
import requests

try:
    import urlparse
except ImportError:
    # pylint: disable=no-name-in-module,import-error
    from urllib import parse as urlparse

from voxcell import VoxelData, Hierarchy, RegionMap

from voxcell.exceptions import VoxcellError


def _download_file(url, filepath, overwrite, allow_empty=False):
    """ Download file from `url` if it is missing. """
    if os.path.exists(filepath) and not overwrite:
        return filepath

    tmp_filepath = filepath + ".download"
    try:
        resp = requests.get(url)
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


class Atlas(object):
    """ Helper class for atlas access. """
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def open(url, cache_dir=None):
        """ Get Atlas object to access atlas stored at URL. """
        parsed = urlparse.urlsplit(url)
        if parsed.scheme in ('', 'file'):
            return LocalAtlas(url)
        elif parsed.scheme in ('http', 'https'):
            if not parsed.path.startswith('/api/analytics/atlas/releases/'):
                raise VoxcellError("Unexpected URL: '%s'" % url)
            if cache_dir is None:
                raise VoxcellError("`cache_dir` should be specified")
            return VoxelBrainAtlas(url, cache_dir)
        else:
            raise VoxcellError("Unexpected URL: '%s'" % url)

    @abc.abstractmethod
    def _fetch_data(self, data_type):
        """ Fetch `data_type` NRRD. """
        pass

    @abc.abstractmethod
    def _fetch_hierarchy(self):
        """ Fetch brain region hierarchy JSON. """
        pass

    def load_data(self, data_type, cls=VoxelData):
        """ Load atlas data layer. """
        return cls.load_nrrd(self._fetch_data(data_type))

    def load_hierarchy(self):
        """ Load brain region hierarchy. """
        return Hierarchy.load_json(self._fetch_hierarchy())

    def load_region_map(self):
        """ Load brain region hierarchy as RegionMap. """
        return RegionMap.load_json(self._fetch_hierarchy())


class VoxelBrainAtlas(Atlas):
    """ Helper class for VoxelBrain atlas. """
    def __init__(self, url, cache_dir):
        self._url = url.rstrip("/")
        resp = requests.get(self._url)
        resp.raise_for_status()
        atlas_id = resp.json()[0]['id']
        assert self._url.endswith(atlas_id)
        self._cache_dir = os.path.join(cache_dir, atlas_id)
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def _fetch_data(self, data_type):
        resp = requests.get(self._url + "/data")
        resp.raise_for_status()
        data_types = []
        for item in resp.json():
            if item['data_type'] == data_type:
                url = item['url']
                break
            data_types.append(item['data_type'])
        else:
            raise VoxcellError(
                "`data_type` should be one of ({0}), provided: {1}".format(
                    ",".join(data_types), data_type
                )
            )
        filepath = os.path.join(self._cache_dir, "%s.nrrd" % data_type)
        return _download_file(url, filepath, overwrite=False)

    def _fetch_hierarchy(self):
        url = self._url + "/filters/brain_region/65535"
        filepath = os.path.join(self._cache_dir, "hierarchy.json")
        return _download_file(url, filepath, overwrite=False)


class LocalAtlas(Atlas):
    """ Helper class for locally stored atlas. """
    def __init__(self, dirpath):
        self.dirpath = dirpath

    def _fetch_data(self, data_type):
        return os.path.join(self.dirpath, "%s.nrrd" % data_type)

    def _fetch_hierarchy(self):
        return os.path.join(self.dirpath, "hierarchy.json")

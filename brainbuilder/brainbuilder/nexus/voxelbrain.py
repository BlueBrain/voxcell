"""
Access to VoxelBrain.

https://nip.humanbrainproject.eu/documentation/user-manual.html#voxel-brain
"""

import os
import requests

from voxcell import VoxelData, Hierarchy


API_ROOT = "http://nip.humanbrainproject.eu/api/analytics/atlas"


def download_file(url, filepath, overwrite):
    """ Download file from `url` if it is missing. """
    if os.path.exists(filepath) and not overwrite:
        return filepath

    tmp_filepath = filepath + ".download"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(tmp_filepath, "wb") as f:
            f.write(resp.content)
        os.rename(tmp_filepath, filepath)
    finally:
        try:
            os.unlink(tmp_filepath)
        except OSError:
            pass

    return filepath


def fetch_data(atlas_id, layer, output_dir, overwrite=False):
    """ Fetch NRRD with `layer` data for given `atlas_id`. """
    params = {
        'atlas_id': atlas_id,
        'layer': layer
    }
    url = API_ROOT + "/download?uri={atlas_id}/{layer}/{layer}.nrrd".format(**params)
    filepath = os.path.join(output_dir, "{atlas_id}-{layer}.nrrd".format(**params))
    return download_file(url, filepath, overwrite=overwrite)


def fetch_hierarchy(atlas_id, output_dir, overwrite=False):
    """ Fetch JSON with brain region hierarchy for given `atlas_id`. """
    params = {
        'atlas_id': atlas_id,
    }
    url = API_ROOT + "/releases/{atlas_id}/filters/brain_region/65535".format(**params)
    filepath = os.path.join(output_dir, "{atlas_id}.json".format(**params))
    return download_file(url, filepath, overwrite=overwrite)


class Atlas(object):
    """ Helper class for atlas access. """
    def __init__(self, atlas_id, cache_dir):
        self._id = atlas_id
        self._cache_dir = cache_dir

    def load_data(self, layer, cls=VoxelData):
        """ Load atlas data layer. """
        if self._id.startswith("/"):
            filepath = os.path.join(self._id, layer + ".nrrd")
        else:
            filepath = fetch_data(self._id, layer, output_dir=self._cache_dir)
        return cls.load_nrrd(filepath)

    def load_hierarchy(self):
        """ Load brain region hierarchy. """
        if self._id.startswith("/"):
            filepath = os.path.join(self._id, "hierarchy.json")
        else:
            filepath = fetch_hierarchy(self._id, output_dir=self._cache_dir)
        return Hierarchy.load_json(filepath)

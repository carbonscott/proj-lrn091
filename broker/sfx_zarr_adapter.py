import zarr
from tiled.adapters.zarr import ZarrArrayAdapter
from tiled.utils import path_from_uri


class SFXZarrAdapter:
    @classmethod
    def from_catalog(cls, data_source, node, /, **kwargs):
        uri = data_source.assets[0].data_uri
        path = path_from_uri(uri)

        dataset = kwargs.get("dataset", "images")
        slice_idx = kwargs.get("slice", None)

        store = zarr.open(path, mode="r")
        array = store[dataset]

        if slice_idx is not None:
            array = array[int(slice_idx)]

        return ZarrArrayAdapter(
            array,
            structure=data_source.structure,
            metadata=node.metadata_,
            specs=node.specs,
        )

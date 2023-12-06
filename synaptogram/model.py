from matplotlib import transforms as T
import numpy as np
import pandas as pd

from atom.api import Atom, Dict, Float, Int, Str, Typed, Value
from raster_geometry import sphere

from ndimage_enaml.model import NDImage
from ndimage_enaml.util import get_image, tile_images


class TiledNDImage(Atom):
    '''
    This duck-types some things in NDImage that allow us to use this with the NDImageView.
    '''
    info = Dict()
    tile_info = Typed(pd.DataFrame)
    tiles = Typed(np.ndarray)
    n_cols = Int(20)
    padding = Int(2)

    sort_channel = Str()
    sort_value = Str()
    sort_radius = Float(0.5)
    ordering = Value()

    labels = Dict()

    def __init__(self, info, tile_info, tiles, **kwargs):
        super().__init__(info=info, tile_info=tile_info, tiles=tiles, **kwargs)

    def get_image(self, *args, **kwargs):
        fn = getattr(np, self.sort_value)
        template = sphere(self.tiles.shape[1:-1], self.sort_radius / self.get_voxel_size('x'))
        if self.sort_channel:
            c = self.channel_names.index(self.sort_channel)
            tiles = self.tiles[..., c] * template
            self.ordering = fn(tiles, axis=(1, 2, 3)).argsort().tolist()
        else:
            tiles = self.tiles * template[..., np.newaxis]
            self.ordering = fn(self.tiles, axis=(1, 2, 3, 4)).argsort().tolist()
        images = get_image(self.tiles, self.channel_names, *args, **kwargs)

        labels = {l: [self.ordering.index(i) for i in s] for l, s in self.labels.items()}
        return tile_images(images[self.ordering], self.n_cols, self.padding, labels)

    @property
    def z_slice_max(self):
        return self.tiles.shape[3]

    @property
    def channel_names(self):
        return [c['name'] for c in self.info['channels']]

    def get_voxel_size(self, dim):
        return self.info['voxel_size']['xyz'.index(dim)]

    def get_image_extent(self):
        n = len(self.tiles)
        n_rows = int(np.ceil(n / self.n_cols))
        xs, ys = self.tiles.shape[1:3]
        x_size = (xs + self.padding) * self.n_cols + self.padding
        y_size = (ys + self.padding) * n_rows + self.padding
        return (0, x_size, 0, y_size)

    def get_image_transform(self):
        return T.Affine2D()

    def tile_index(self, x, y):
        xs, ys = self.tiles.shape[1:3]
        xs += self.padding
        ys += self.padding
        xi = (x - self.padding) // xs
        yi = (y - self.padding) // ys
        if (xi < 0) or (yi < 0):
            return -1
        if (i := yi * self.n_cols + xi) < len(self.tiles):
            i = int(i)
            return self.ordering[i]
        return -1

    def select_next_tile(self, i, step):
        j = self.ordering.index(i) + step
        if not (0 <= j < len(self.ordering)):
            return self._select_tile(i)
        return self._select_tile(self.ordering[j])

    def label_tile(self, i, label):
        if i == -1:
            return
        self.labels.setdefault(label, set()).add(i)

    def unlabel_tile(self, i, label=None):
        if i == -1:
            return
        if label is None:
            for l, indices in self.labels.items():
                if l == 'selected':
                    continue
                if i in indices:
                    indices.remove(i)

    def select_tile_by_coords(self, x, y):
        i = self.tile_index(x, y)
        if i == -1:
            return
        return self._select_tile(i)

    def _select_tile(self,  i):
        self.labels['selected'] = set([i])
        result = self.tile_info.iloc[i].to_dict()
        result['i'] = i
        return result


class Points(Atom):

    overview = Typed(NDImage)
    points = Typed(TiledNDImage)

    def __init__(self, image_info, image, point_info, point_images):
        self.overview = NDImage(image_info, image)
        self.points = TiledNDImage(image_info, point_info, point_images)

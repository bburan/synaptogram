from matplotlib import transforms as T
import numpy as np
import pandas as pd

from atom.api import Atom, Dict, Int, Str, Typed, Value

from ndimage_enaml.model import NDImage
from ndimage_enaml.util import get_image, tile_images


class TiledNDImage(Atom):

    info = Dict()
    tile_info = Typed(pd.DataFrame)
    tiles = Typed(np.ndarray)
    n_cols = Int(20)
    padding = Int(1)
    sort_channel = Str()
    sort_value = Str()
    ordering = Value()

    def __init__(self, info, tile_info, tiles, **kwargs):
        super().__init__(info=info, tile_info=tile_info, tiles=tiles, **kwargs)

    def get_image(self, *args, **kwargs):
        print('getting image')
        fn = getattr(np, self.sort_value)
        if self.sort_channel:
            c = self.channel_names.index(self.sort_channel)
            self.ordering = fn(self.tiles[..., c], axis=(1, 2, 3)).argsort()
        else:
            self.ordering = fn(self.tiles, axis=(1, 2, 3, 4)).argsort()
        images = get_image(self.tiles, self.channel_names, *args, **kwargs)
        return tile_images(images[self.ordering], self.n_cols, self.padding)

    @property
    def z_slice_max(self):
        return self.tiles.shape[3]

    @property
    def channel_names(self):
        return [c['name'] for c in self.info['channels']]

    def get_voxel_size(self, dim):
        return self.info['voxel_size']['xyz'.index(dim)]

    def get_voxel_size(self, dim):
        return 1

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


class Points(Atom):

    overview = Typed(NDImage)
    points = Typed(TiledNDImage)

    def __init__(self, image_info, image, point_info, point_images):
        self.overview = NDImage(image_info, image)
        self.points = TiledNDImage(image_info, point_info, point_images)

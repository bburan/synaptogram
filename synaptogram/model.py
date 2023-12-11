from matplotlib import transforms as T
import numpy as np
import pandas as pd

from atom.api import Atom, Dict, Enum, Event, Float, Int, Str, Typed, Value, observe
from raster_geometry import sphere

from ndimage_enaml.model import get_channel_config, make_channel_config, NDImage
from ndimage_enaml.util import get_image, tile_images


class TiledNDImage(Atom):
    '''
    This duck-types some things in NDImage that allow us to use this with the NDImageView.
    '''
    info = Dict()
    tile_info = Typed(pd.DataFrame)
    tiles = Typed(np.ndarray)
    n_cols = Int(12)
    padding = Int(3)

    sort_channel = Str()
    sort_value = Enum('max', 'mean', 'median')
    sort_radius = Float(0.5)
    ordering = Value()

    labels = Dict()
    channel_config = Dict()
    labels_updated = Event()

    def __init__(self, info, tile_info, tiles, **kwargs):
        super().__init__(info=info, tile_info=tile_info, tiles=tiles, **kwargs)
        self.channel_config = make_channel_config(info)
        self._update_ordering()

    @observe('sort_channel', 'sort_value', 'sort_radius')
    def _update_ordering(self, event=None):
        template = sphere(self.tiles.shape[1:-1], self.sort_radius / self.get_voxel_size('x'))
        fn = getattr(np, self.sort_value)
        if self.sort_channel:
            c = self.channel_names.index(self.sort_channel)
            tiles = self.tiles[..., c] * template
            self.ordering = fn(tiles, axis=(1, 2, 3)).argsort().tolist()
        else:
            tiles = self.tiles * template[..., np.newaxis]
            self.ordering = fn(tiles, axis=(1, 2, 3, 4)).argsort().tolist()

    def get_channel_config(self, channels=None):
        if channels is None:
            channels = self.channel_names
        return get_channel_config(channels, self.channel_config)

    def get_image(self, channels, *args, **kwargs):
        channel_config = self.get_channel_config(channels)
        images = get_image(self.tiles, channel_config, *args, **kwargs)
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
        self.labels.setdefault(label, {})[i] = self.tile_info.iloc[i].to_dict()
        self.labels_updated = True

    def unlabel_tile(self, i, label=None):
        if i == -1:
            return
        if label is None:
            for l, indices in self.labels.items():
                if l == 'selected':
                    continue
                if i in indices:
                    del indices[i]
        else:
            if i in self.labels[label]:
                del self.labels[i]
        self.labels_updated = True

    def select_tile_by_coords(self, x, y):
        i = self.tile_index(x, y)
        if i == -1:
            return
        return self._select_tile(i)

    def _select_tile(self,  i):
        info = self.tile_info.iloc[i].to_dict()
        self.labels['selected'] = {i: info}
        info['i'] = i
        return info

    def get_state(self):
        labels = {l: i for l, i in self.labels.items() if l != 'selected'}
        return {
            'labels': labels,
        }

    def set_state(self, state):
        for label, indices in state['labels'].items():
            indices = {int(k): v for k, v in indices.items()}
            self.labels[label] = indices


class Points(Atom):

    overview = Typed(NDImage)
    points = Typed(TiledNDImage)

    def __init__(self, image_info, image, point_info, point_images):
        self.overview = NDImage(image_info, image)
        self.points = TiledNDImage(image_info, point_info, point_images)

    def get_state(self):
        return {
            'overview': self.overview.get_state(),
            'points': self.points.get_state(),
        }

    def set_state(self, state):
        self.overview.set_state(state['overview'])
        self.points.set_state(state['points'])

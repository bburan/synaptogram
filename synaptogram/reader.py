from functools import cached_property

import numpy as np
import pandas as pd

import tables as tb

from .model import Points

def extract_value(attrs, key):
    return float(''.join(attrs[key].astype('U')))


def extract_points(node):
    cols = []
    if 'CoordsXYZR' in node:
        coords = node.CoordsXYZR.read()
        cols.extend(['x', 'y', 'z', 'radius_x'])
    else:
        coords = []
    if 'RadiusYZ' in node:
        radius = node.RadiusYZ.read()
        cols.extend(['radius_y', 'radius_z'])
        coords =  np.c_[coords, node.RadiusYZ.read()]
    return pd.DataFrame(coords, columns=cols)


class BaseReader:

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        marker = 'CtBP2'
        return Points(
                self.image_info,
                self.image,
                self.points.xs(marker, level='marker'),
                self.get_point_volumes(marker),
                )


class BaseImarisReader(BaseReader):

    def __init__(self, filename):
        self.filename = filename
        self.fh = tb.open_file(filename)

    @cached_property
    def points(self):
        points = []
        keys = []
        for i, (name, node) in enumerate(self.fh.root.Scene.Content._v_children.items()):
            if name.startswith('Points'):
                p = extract_points(node)
                points.append(p)
                keys.append((i, node._v_attrs['Name'][0].decode('utf')))
        points = pd.concat(points, keys=keys, names=['node_index', 'marker', 'i'])
        for d, dim in enumerate('xyz'):
            origin = self.image_info['lower'][d]
            size = self.image_info['voxel_size'][d]
            i = (points[dim] - origin) / size
            points[f'{dim}i'] = i.round().astype('i')
        return points

    @cached_property
    def image_info(self):
        image_attrs = self.fh.root.DataSetInfo.Image._v_attrs
        xlb = extract_value(image_attrs, 'ExtMin0')
        ylb = extract_value(image_attrs, 'ExtMin1')
        zlb = extract_value(image_attrs, 'ExtMin2')
        xub = extract_value(image_attrs, 'ExtMax0')
        yub = extract_value(image_attrs, 'ExtMax1')
        zub = extract_value(image_attrs, 'ExtMax2')
        xvoxels = extract_value(image_attrs, 'X')
        yvoxels = extract_value(image_attrs, 'Y')
        zvoxels = extract_value(image_attrs, 'Z')
        return {
            'lower': [xlb, ylb, zlb],
            'n_voxels': [
                int(xvoxels),
                int(yvoxels),
                int(zvoxels),
            ],
            'voxel_size': [
                np.abs(xub-xlb) / xvoxels,
                np.abs(yub-ylb) / yvoxels,
                np.abs(zub-zlb) / zvoxels,
            ],
            'channels': [
                {'name': 'CtBP2'},
                {'name': 'GluR2'},
                {'name': 'MyosinVIIa'},
            ],
        }

    @cached_property
    def image(self):
        img_node = self.fh.root.DataSet._f_get_child('ResolutionLevel 0/TimePoint 0')
        data = []
        for channel_node in img_node._f_iter_nodes():
            d = channel_node.Data[:]
            data.append(d[..., np.newaxis])
        data = np.concatenate(data, axis=-1)
        x, y, z = self.image_info['n_voxels']
        return data[:z, :y, :x].swapaxes(0, 2)

    def get_point_volumes(self, point_name, size=10):
        spots = []
        xyz_max = self.image.shape[:-1]
        points = self.points.xs(point_name, level='marker')
        volumes = []
        for _, point in points.iterrows():
            i = point[['xi', 'yi', 'zi']].abs().astype('i').values
            lb = np.clip(i - 5, 0, xyz_max)
            ub = np.clip(i + 5, 0, xyz_max)
            s = np.s_[*[slice(l, u) for l, u in zip(lb, ub)]]
            image = self.image[s]

            padding = size - np.array(image.shape[:-1])
            if np.any(padding):
                padding = list(zip(np.zeros_like(padding), padding)) + [(0, 0)]
                image = np.pad(image, padding)
            volumes.append(image[np.newaxis])
        return np.concatenate(volumes, axis=0)


class ImarisReader(BaseImarisReader):

    @cached_property
    def points(self):
        points = super().points
        names = points.index.names[:]
        points = points.reset_index()
        points['marker'].map(lambda x: 'CtBP2' if x == 'Spots 1' else x)
        return points.set_index(names)

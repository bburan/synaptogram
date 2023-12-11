from copy import deepcopy

from atom.api import Atom, Bool, Dict, Event, Float, Instance, Int, Str, Typed, Value
from enaml.application import deferred_call
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
import numpy as np

from ndimage_enaml.model import NDImageCollection
from ndimage_enaml.util import project_image
from ndimage_enaml.presenter import FigurePresenter, NDImageCollectionPresenter, NDImagePlot, StatePersistenceMixin

from .model import Points, TiledNDImage
from .reader import BaseReader


class OverviewPresenter(NDImageCollectionPresenter):

    highlight_artist = Value()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highlight_artist = Circle((0, 0), radius=0, linewidth=1, facecolor='none', edgecolor='white')
        self.axes.add_patch(self.highlight_artist)
        self.current_artist.display_mode = 'slice'
        self.current_artist.z_slice_thickness = 10

    def highlight_selected(self, event):
        value = event['value']
        if not value:
            return
        span = 6
        extent = (
                value['x'] - span,
                value['x'] + span,
                value['y'] - span,
                value['y'] + span,
                )
        self.axes.axis(extent)
        self.highlight_artist.set_center((value['x'], value['y']))
        self.highlight_artist.set_radius(0.5)
        self.current_artist.center_z_substack(int(value['zi']))
        self.request_redraw()


class TiledNDImagePlot(NDImagePlot):

    sort_channel = Str('GluR2')
    sort_value = Str('max')
    sort_radius = Float(0.5)

    def _observe_sort_radius(self, event):
        self.ndimage.sort_radius = self.sort_radius
        self.request_redraw()

    def _observe_sort_channel(self, event):
        self.ndimage.sort_channel = self.sort_channel
        self.request_redraw()

    def _observe_sort_value(self, event):
        self.ndimage.sort_value = self.sort_value
        self.request_redraw()


class PointsPresenter(NDImageCollectionPresenter):

    obj = Instance(TiledNDImage)
    artist = Value()
    selected = Dict()
    selected_coords = Value()

    def _default_artist(self):
        artist = TiledNDImagePlot(self.axes)
        artist.observe('updated', self.request_redraw)
        return artist

    @property
    def current_artist(self):
        return self.artist

    def _observe_obj(self, event):
        self.artist.ndimage = self.obj

    def right_button_press(self, event):
        x, y = event.xdata, event.ydata
        self.selected = self.obj.select_tile_by_coords(x, y)
        self.request_redraw()

    def key_press(self, event):
        if event.key.lower() == 'd':
            self.apply_label('artifact')
        if event.key.lower() == 'o':
            self.apply_label('orphan')
        if event.key.lower() == 'c':
            self.clear_label()
        if event.key.lower() == 'right':
            self.select_next_tile(1)
        if event.key.lower() == 'left':
            self.select_next_tile(-1)
        if event.key.lower() == 'up':
            self.select_next_tile(self.obj.n_cols)
        if event.key.lower() == 'down':
            self.select_next_tile(-self.obj.n_cols)

    def apply_label(self, label):
        self.obj.unlabel_tile(self.selected['i'])
        self.obj.label_tile(self.selected['i'], label)
        self.request_redraw()

    def clear_label(self):
        self.obj.unlabel_tile(self.selected['i'])
        self.request_redraw()

    def select_next_tile(self, step):
        with self.suppress_notifications():
            if step is None:
                i = self.obj.ordering[0]
                step = 0
            else:
                i = self.selected.get('i', self.obj.ordering[0])
        self.selected = self.obj.select_next_tile(i, step)
        self.request_redraw()

    def redraw(self):
        self.artist.redraw()
        super().redraw()

    def check_for_changes(self):
        pass


class PointProjectionPresenter(FigurePresenter):

    obj = Value()
    artist = Value()
    vertical_crosshairs = Value()
    horizontal_crosshairs = Value()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axes.set_axis_off()
        self.artist = AxesImage(self.axes, data=np.array([[]]), origin='lower')
        self.axes.add_artist(self.artist)
        self.axes.axis('equal')
        self.vertical_crosshairs = [self.axes.axvline(0, color='w', ls=':', alpha=0.5) for _ in range(6)]
        self.horizontal_crosshairs = [self.axes.axhline(0, color='w', ls=':', alpha=0.5) for _ in range(2)]

    def highlight_selected(self, event):
        padding = 1
        tile = self.obj.tiles[event['value']['i']]
        img = project_image(tile, self.obj.get_channel_config(), padding)
        self.artist.set_data(img)
        y, x = img.shape[:2]
        self.artist.set_extent((0, x, 0, y))

        xs, ys, _, _ = tile.shape
        for i, a in enumerate(self.vertical_crosshairs):
            o = i * (xs + padding) + padding + xs * 0.5
            a.set_data(([o, o], [0, 1]))
        for i, a in enumerate(self.horizontal_crosshairs):
            o = i * (xs + padding) + padding + ys * 0.5
            a.set_data(([0, 1], [o, o]))

        self.figure.canvas.draw()


class SynaptogramPresenter(StatePersistenceMixin):

    obj = Typed(object)
    reader = Instance(BaseReader)
    overview = Instance(OverviewPresenter)
    points = Instance(PointsPresenter)
    point_projection = Instance(PointProjectionPresenter)

    def _observe_obj(self, event):
        if self.obj is not None:
            self.overview = OverviewPresenter(obj=NDImageCollection([self.obj.overview]))
            self.point_projection = PointProjectionPresenter(obj=self.obj.points)
            self.points = PointsPresenter(obj=self.obj.points)
            self.points.observe('selected', self.overview.highlight_selected)
            self.points.observe('selected', self.point_projection.highlight_selected)

            deferred_call(self.points.select_next_tile, None)
            self.obj.points.observe('labels_updated', self.check_for_changes)

    def update_state(self):
        super().update_state()
        self.points.request_redraw()

    def check_for_changes(self, event=None):
        saved = self.saved_state['data']['points']['labels']
        unsaved = self.get_full_state()['data']['points']['labels']
        saved.pop('selected', None)
        unsaved.pop('selected', None)
        get_labels = lambda s: {k: list(int(i) for i in v) for k, v in s.items() if len(v) > 0}
        self.unsaved_changes = get_labels(saved) != get_labels(unsaved)

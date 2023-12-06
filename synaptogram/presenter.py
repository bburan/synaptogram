from atom.api import Atom, Bool, Event, Float, Instance, Int, Str, Typed, Value
from enaml.application import deferred_call
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
import numpy as np

from ndimage_enaml.model import NDImageCollection
from ndimage_enaml.util import project_image
from ndimage_enaml.presenter import FigurePresenter, NDImageCollectionPresenter, NDImagePlot

from .model import Points, TiledNDImage


class OverviewPresenter(NDImageCollectionPresenter):

    highlight_artist = Value()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highlight_artist = Circle((0, 0), radius=0, linewidth=1, facecolor='none', edgecolor='white')
        self.axes.add_patch(self.highlight_artist)

    def highlight_selected(self, event):
        value = event['value']
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
    selected = Value()
    selected_coords = Value()

    @property
    def current_artist(self):
        return self.artist

    def _observe_obj(self, event):
        self.artist = TiledNDImagePlot(self.axes, self.obj)
        self.artist.observe('updated', self.update)
        self.axes.axis('equal')
        self.axes.axis(self.obj.get_image_extent())

    def right_button_press(self, event):
        x, y = event.xdata, event.ydata
        self.selected = self.obj.select_tile_by_coords(x, y)
        self.artist.request_redraw()

    def key_press(self, event):
        if event.key.lower() == 'd':
            self.obj.label_tile(self.selected['i'], 'artifact')
        if event.key.lower() == 'o':
            self.obj.label_tile(self.selected['i'], 'orphan')
        if event.key.lower() == 'c':
            self.obj.unlabel_tile(self.selected['i'])
        if event.key.lower() == 'right':
            self.selected = self.obj.select_next_tile(self.selected['i'], 1)
        if event.key.lower() == 'left':
            self.selected = self.obj.select_next_tile(self.selected['i'], -1)
        if event.key.lower() == 'up':
            self.selected = self.obj.select_next_tile(self.selected['i'], self.obj.n_cols)
        if event.key.lower() == 'down':
            self.selected = self.obj.select_next_tile(self.selected['i'], -self.obj.n_cols)
        self.artist.request_redraw()

    def check_for_changes(self):
        pass


class PointProjectionPresenter(FigurePresenter):

    obj = Value()
    artist = Value()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axes.set_axis_off()
        self.artist = AxesImage(self.axes, data=np.array([[]]), origin='lower')
        self.axes.add_artist(self.artist)

    def highlight_selected(self, event):
        tile = self.obj.tiles[event['value']['i']]
        img = project_image(tile, self.obj.channel_names)
        self.artist.set_data(img)
        y, x = img.shape[:2]
        self.artist.set_extent((0, x, 0, y))
        self.figure.canvas.draw()


class SynaptogramPresenter(Atom):

    obj = Typed(object)
    overview = Instance(OverviewPresenter, {})
    points = Instance(PointsPresenter, {})
    point_projection = Instance(PointProjectionPresenter, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points.observe('selected', self.overview.highlight_selected)
        self.points.observe('selected', self.point_projection.highlight_selected)

    def _observe_obj(self, event):
        if self.obj is not None:
            self.overview.obj = NDImageCollection([self.obj.overview])
            self.points.obj = self.obj.points
            self.point_projection.obj = self.obj.points

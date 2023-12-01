from atom.api import Atom, Bool, Event, Instance, Int, Typed, Value
from enaml.application import deferred_call
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
import numpy as np

from ndimage_enaml.model import NDImageCollection
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
        span = 4
        extent = (
                value['x'] - span,
                value['x'] + span,
                value['y'] - span,
                value['y'] + span,
                )
        self.axes.axis(extent)
        self.highlight_artist.set_center((value['x'], value['y']))
        self.highlight_artist.set_radius(1)
        self.request_redraw()


class PointsPresenter(FigurePresenter):

    obj = Instance(TiledNDImage)
    artist = Value()
    selected = Value()

    def _observe_obj(self, event):
        self.artist = NDImagePlot(self.axes, self.obj)
        self.artist.observe('updated', self.update)
        self.axes.axis('equal')
        self.axes.axis(self.obj.get_image_extent())

    def left_button_press(self, event):
        x, y = event.xdata, event.ydata
        i = self.obj.tile_index(x, y)
        self.selected = self.obj.tile_info.iloc[i].to_dict()

    def check_for_changes(self):
        pass


class SynaptogramPresenter(Atom):

    obj = Typed(object)
    overview = Instance(OverviewPresenter, {})
    points = Instance(PointsPresenter, {})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points.observe('selected', self.overview.highlight_selected)


    def _observe_obj(self, event):
        self.overview.obj = NDImageCollection([self.obj.overview])
        self.points.obj = self.obj.points

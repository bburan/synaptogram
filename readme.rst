Synaptogram
===========

Introduction
------------

This facilitates screening synaptic ribbons identified in Imaris to determine
if they are artifacts or orphans (i.e., without a post-synaptic bouton). This
program will work with any Imaris file that contains the `Spots` feature named
`CtBP2`.

Using the program
-----------------

Three visualizations are provided:

* Overview image (upper left). When a ribbon is selected, the overview image
  updates to show the location of the synaptic ribbon (marked with a white
  circle) in the confocal image. This context can be useful for determining the
  status of a ribbon.
* Projection of the selected ribbon (lower left). Each channel (GluR2, CtBP2,
  and Myosin VIIa) are shown separately as maximum projections onto the XY, YZ
  and XZ planes. The crosshair shows the center of the ribbon in each image.
  Sometimes other ribbons will appear in the same image, so focus on the
  crosshair.
* Tiled overview of all ribbons identified by Imaris. The currently selected
  ribbon is indicated by a white square. Artifacts are indicated by red squares
  and orphans by green squares.

Ribbons in the tiled image are sorted by default based on the maximum intensity
of the GluR2 label. If needed, you can sort by other criteria. The goal is to
identify all ribbons that are artifacts (e.g., in the nucleus or outside the
cytoplasm of the inner hair cell) or orphans (i.e., no post-synaptic receptor
associated with the ribbon).

Mouse interaction
.................
left click + drag (overview and tiled ribbons)
    Pan image
right click (tiled ribbons visualization only)
    Select select ribbon for analysis.
mouse wheel (overview and tiled ribbons)
    Zoom in/out

Keyboard shortcuts
..................
These keyboard shortcuts only work when the tiled visualization has focus
(e.g., by left-clicking on it). The focus is set to the tiled visualization by
default, but can be lost if you click on one of the other visualizations in the
window.

a or d
    Mark ribbon as artifact
o
    Mark ribbon as orphan
c
    Clear label
arrow keys
    Navigate through the selected ribbons with the arrow keys. Scrolling off
    the end of one row will take you to the next row.
Ctrl + S
    Save the analysis

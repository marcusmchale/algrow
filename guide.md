# Algrow GUI Guide
When AlGrow is loaded, a menu is presented along with a logo page.
Chose from the options described below to proceed.

## File
The file menu contains options to:
  - [load an image](#load-image) for configuration,
  - optionally [load an image mask](#load-mask) for additional features during target hull definition,
  - [load a previously saved configuration](#load-configuration) from a text file,
  - [save the current configuration](#save-configuration) to a text file, or
  - Quit the application.

### Load Image
Load an RGB image for [configuration](#configuration). The current loaded image filename is displayed 
in the bottom left of the relevant configuration toolbar.
  - Select a supported RGB image file (.jpeg, .jpg, .png or .bmp extensions).
### Load Mask
Load an image mask for the current loaded image. 
  - Select a supported boolean or greyscale image file (.jpeg, .jpg, .png or .bmp extensions).
  - A mask provides two additional facilities during [target colour](#target-colour) specification:
    - the [dice-coefficient](#dice-coefficient) and
    - construction of a target hull [from the mask](#from-mask).

### Load Configuration
Load settings for configuration or analysis from a previously saved configuration file
  - Select an algrow configuration file (.conf) to load.
### Save Configuration
Save the current settings to a configuration file for future use, either in the GUI or in the CLI. 
  - Select a path and enter a filename to save an algrow configuration file (.conf).

## Configuration
The configuration menu conotains options to:
  - Set the image [scale](#scale),
  - Defined the [target circle colour](#circle-detection),
  - Define the [layout detection parameters](#circle-layout), or 
  - Define the [target colour](#target-colour) for image segmentation

### Scale
An internal reference with known dimensions should be included in at least one image at the relevant image depth.
This tool allows you to use this internal reference to calculate an appropriate scale
to convert from pixels to units of physical distance and area (mm²).

  - Shift-Click once to start drawing a line and again to finish the line. This will provide a value in pixels (px)
to the toolbar. 
  - Type the physical distance (mm) for this line into the next box and press enter to calculate a scale (px/mm).
  - It may help to zoom in on the image to accurately define the start and end of the line. Use a mouse wheel to zoom in and out.
  - You can change the colour of the line drawn using the line colour dialog.
  - Alternatively you can enter a scale manually into the box and press enter.

### Circle detection
To dynamically detect a layout, AlGrow first detects circular edges in a grayscale image. 
This grayscale image is constructed from the distance (in Lab colourspace) from a chosen circle colour.
The circle colour should be chosen to create a high contrast edge in a circular pattern surrounding each subject.  

  - Shift-click on the loaded image to select a pixel colour.
    - You may wish to zoom in to select a small part of the image using the mouse wheel.
    - Once at least one pixel colour has been selected, the greyscale image will be presented.
    - Clicking on the greyscale image will select the colour of the corresponding pixel in the source image.
  - Each selected pixel colour is displayed as a button in the toolbar.
    - Individual colours can be removed from the selection by clicking on its corresponding button.
    - All selected colours can be removed with the "clear" button. 
  - The average colour (median in each dimension of Lab) is displayed on the "circle colour" button.
    - This average colour is used to construct the contrast grayscale image.
      - The contrast image is displayed when at least one colour is selected
  - Configuration is complete when a strong contrast "edge" is visible for each target circle.

### Circle Layout
To dynamically detect a layout, Algrow needs additional parameters to: 
identify target circles, cluster circles into plates/trays and assign these incrementing integer identities.

Three of these can be measured from a loaded image in this interface, as is done with the [scale](#Scale) utility:
  - [circle diameter](#circle-diameter),
  - [circle separation](#circle-separation) and
  - [plate width](#plate-width).

There are then two counts that must be supplied, and be accurate for all images: 
  - the number of [circles](#circles) per plate
  - and the number of [plates](#plates).

Three tolerance factors are also provided to account for:
  - [circle variability](#circle-variability),
  - [circle expansion](#circle-expansion) and
  - [circle separation tolerance](#circle-separation-tolerance).

Additionally, there are six toggle buttons affecting the direction of ID incrementation, across plates:
  - [plates in rows/plates in columns](#plates-in-rows),
  - [plates start left/plates start right](#plates-start-right) and 
  - [plates start top/plates start bottom](#plates-start-top), 

and within plates:
  - [circles in rows/circles in columns](#circles-in-rows),
  - [circles start left/circles start right](#circles-start-right) and 
  - [circles start top/circles start bottom](#circles-start-top).

There are then buttons to:
  - [detect a defined layout](#detect-layout) using the supplied parameters,
  - [save a fixed layout](#save-fixed-layout) to a text file, or to  
  - [clear a detected layout](#clear-layout).

#### Circle diameter
To detect circles in the [circle colour distance image](#circle-detection), 
the diameter is provided to determine the center of a range of radii to scan in a Hough transform.
The breadth of the range to scan can be adjusted with the [circle variability](#circle-variability) tolerance factor

  - Draw a line between far edges of a target circle 
  - Copy the value from "px" to "circle diameter"

#### Circle separation
To cluster circles into plates, a dendrogram is constructed reflecting the distance between circle centers.
A key assumption to this plate identification strategy is that the distances between circles is less within plates 
than it is between plates. If this assumption is not held, consider treating the layout as a single plate.

The circle distance dendrogram is cut to identify clusters of a size 
corresponding to the specified number of [circles per plate](#circles).
The cut height is determined by the distance between circle centers, 
which is calculated from the supplied [circle diameter](#circle-diameter) 
and the value for separation between circle edges.
An additional [circle separation](#circle-separation) tolerance factor is provided,
with higher values increasing the cut height.

Once plates have been identified, their alignment to the image axes is assessed 
by examining the angles between up to three points on each of the four possible corners.
The position of each circle within the plate is then adjusted according to a corrective rotation
to the image axes. 

Once this rotation correction is applied, 
further clustering is performed into either [rows or columns](#circles-in-rows).
This strategy provides a robust mechanism to identify grid-like patterns of experimental units. 
The relative positions along the relevant axis for each row or column (averaged across members of the row/column),
and the relative position of circles within each row/column, defines the incrementation order,
along with two additional circle id incrementation instructions; to 
start from the [left or right](#circles-start-left) and to start at the [top or bottom](#circles-start-top).

  - Draw a line between edges of adjacent target circles within a plate
  - Copy the value from "px" to "circle separation"

#### Plate width
Once clusters of circles conforming to the plate specification have been identified,
the centers of these plates can then also be clustered into either [rows or columns](#plates-in-rows).
A dendrogram of the distances between plates along the relevant axis is cut to define rows/columns,
with the cut height is defined as half of the plate width.
This strategy provides limited tolerance, scaling with plate size, 
to imperfect alignment of plates into rows/columns along the relevant image axis.
The relative positions along the relevant axis for each row or column (averaged across members of the row/column),
and the relative positions of plates within each row/column, defines the incrementation order, 
along with two additional plate id incrementation instructions; to 
start from the [left or right](#plates-start-left) and to start at the [top or bottom](#plates-start-top).

  - Draw a line across the shortest axis of a plate
  - Copy the value from "px" to "plate width"

#### Counts
##### Circles
Plates within a layout specification must have a constant number of target circles.
This number is defined in the "circles" input field.
  - Enter the number of circles per plate
##### Plates
Each set of images sharing a layout specification must have a constant number of plates.
This number is defined in the "plates" input field.
  - Enter the number of plates per image

#### Tolerance factors
##### Circle variability
This factor affects the range of radii to search during circle detection.
  - A higher number will search for more radii:
    - e.g. with a [circle diameter](#circle-diameter) of 100 and a circle variability of 0.1:
      - radii from 45 to 55 px will be considered.
  - The number of radii considered in the provided range is currently a constant.
    - n=11, i.e. 5 above and 5 below the defined radius.
    - e.g. with the above example these radii are considered:
      - 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
##### Circle expansion
This factor affects the evaluated target area following circle detection.
  - e.g. With a [circle diameter](#circle-diameter) for detection of 100 px and a circle expansion of 0.1,
the final target area considered will have a diameter of 110 px.
##### Circle separation tolerance
This factor affects the cut-height during [plate detection](#circle-separation).
  - e.g. with a [circle diameter](#circle-diameter) of 100, [circle separation](#circle-separation) of 40, 
the distance between circle centers is thought to be 140. 
With a circle separation tolerance of 0.1, the cut height will be (140 * 1.1) = 154.

#### Plate ID incrementation
Plates are assigned a unique integer per layout and sorted prior to [circle ID incrementation](#circle-id-incrementation).
##### Plates in rows
When toggled, this button will read "plates in columns".
This option affects whether to cluster plates into rows (along the vertical axis)
or columns (along the horizontal axis).
##### Plates start left
When toggled, this button will read "plates start right".
This option affects whether to start plate ID incrementation from the left or right,
and then proceed to the right or left, respectively.
##### Plates start top
When toggled, this button will read "plates start bottom".
This option affects whether to start plate ID incrementation from the top or bottom,
and then proceed to the bottom or top, respectively.

#### Circle ID incrementation
Although circle unit IDs are assigned to be unique per layout, 
these are assigned by incrementation AFTER [sorting plates](#plate-id-incrementation).
##### Circles in rows
When toggled, this button will read "circles in columns".
This option affects whether to cluster circles within plates into rows (along the vertical axis)
or columns (along the horizontal axis).
##### Circles start left
When toggled, this button will read "circles start right".
This option affects whether to start circle ID incrementation from the left or right
(within plates) and then proceed to the right or left, respectively.
##### Circles start top
When toggled, this button will read "circles start bottom".
This option affects whether to start circle ID incrementation from the top or bottom
(within plates) and then proceed to the bottom or top, respectively.

#### Detect layout
This button will start layout detection with the defined parameters.
The duration of this search varies with many factors (from around second up to minute),
so please be patient.

When the search is successful, and a layout is defined, an overlay will be depicted.
This will highlight the detected circles with the applied [circle expansion](#circle-expansion).
It will also overlay the [plate IDs](#plate-id-incrementation) and [circle IDs](#circle-id-incrementation).
If the incrementation options are toggled, some of the search/clustering will be repeated.
If the other parameters are changed you should click the "detect layout" button again
to ensure the parameters result in a successful layout detection.

The layout detection can fail for a number of reasons and a dialog should appear
warning of the issue. Click OK to close the dialogue. 
An overlay is also presented with the detected circles (including the applied [circle expansion](#circle-expansion)).
If sufficient circles were detected, a dendrogram is also presented constructed from the distances between centers.

  - Insufficient circles detected:
    - Examine the detected circles and consider checking and/or adjusting:
      - the [circle colour](#circle-detection) definition,
      - the [circle diameter](#circle-diameter),
      - and [circles](#circles) per plate.
  - Insufficient/Excess plates detected
    - Examine the detected circles image and consider the above listed parameters.
    - Examine the dendrogram and consider checking and/or adjusting:
      - the [circle separation](#circle-separation),
      - the [circle separation tolerance](#circle-separation-tolerance),
      - the [plates](#plates) per image.

#### Save fixed layout
When a layout is detected, it is possible to save this as a fixed layout file (.csv).
This file is a specification of the:
  - plate ID,
  - plate center coordinates,
  - circle ID,
  - circle center coordinates and
  - circle radius.

This file can be modified and custom versions loaded on the [area analysis](#area) page.
When saved or loaded, the fixed layout will be displayed
on both the [layout](#circle-layout) and [target colour](#target-colour) configuration pages.

However, be mindful that such fixed layouts will not support movement.

#### Clear layout
When the fixed-layout is saved, if you wish to return to dynamic layout detection, click "clear layout".
You may then click [detect layout](#detect-layout) again to ensure the parameters are valid 
and load a layout for [target colour](#target-colour) configuration pages.

You may also wish to clear the detected layout
if you would prefer to examine the full image in [target colour](#target-colour) configuration.

### Target colour



If a [layout](#circle-layout) is defined, then the background will be masked from the displayed image
and voxels will not be formed from these regions.




#### Alpha
The alpha parameter is necessary to construct a concave hull, 
which may be necessary when similar background colours are present in the image.
When alpha is 0, the convex hull is constructed. 
When alpha is high, isolated points may be excluded.


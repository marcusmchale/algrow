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
    - construction of a target hull [from the mask](#hull-from-mask).

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
  - When more than once colour is selected, the average colour (median in each dimension of Lab)
is displayed on the "circle colour" button.
    - This colour is used to construct the contrast grayscale image.
  - Configuration is complete when a strong contrast circular "edge" is visible for each target region.

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
  - [plates start left/plates start right](#plates-start-left) and 
  - [plates start top/plates start bottom](#plates-start-top), 

and within plates:
  - [circles in rows/circles in columns](#circles-in-rows),
  - [circles start left/circles start right](#circles-start-left) and 
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
If a layout has been detected or [loaded](#fixed-layout), it can be removed by clicking "clear layout".
This may be desired if you would prefer to examine the full image in [target colour](#target-colour) configuration,
or to return to dynamic layout detection after loading a fixed layout.

### Target colour
The target colour configuration panel consists of a
3D plot (top-left), 
a toolbar (bottom left)
and an image preview (right).

The 3D-plot is in the Lab colourspace, 
with points representing a down-sampling of image pixel values into [voxels](#voxel-size).
The view can be rotated by left-click and drag, 
panned by a right-click and drag,
or zoomed with a mouse wheel. 
Voxels can be selected by shift-clicking on the points, 
with the nearest point to the observer in a given radius of the click being selected.
When selected, a voxel is displayed as a sphere, 
and the corresponding voxel colour is displayed as a button under "selected" in the toolbar.
Selected voxels can be de-selected either by shift-clicking on the sphere, 
or by clicking on the corresponding button.
When sufficient voxels are selected, a [hull](#show-hull) will be displayed.

The image preview highlights pixels corresponding to [selected](#show-selected) voxels, 
and [target](#show-target) pixels, i.e. those corresponding to voxels inside 
or within [delta](#delta) of the hull surface.
Voxels can also be selected/deselected by shift clicking on pixels in this image preview,
and the view can be zoomed with a mouse wheel, or panned by clicking and dragging.

If a layout has been [detected](#detect-layout) or [loaded](#load-fixed-layout), 
the masked pixels are not considered for voxel sampling and are masked in black in the image preview.

Each of the inputs and buttons in the toolbar, and their impact 
on the 3D plot and image preview are described below.

#### Min Pixels
Displayed voxels can be filtered by the minimum number of pixels represented by each voxel.
Raising this value is useful to highlight clusters of pixel colour.

The "min pixels" value also affects hull construction [from a loaded mask](#from-mask).

#### Alpha
The alpha parameter guides [hull](#show-hull) construction from the selected points.
When set to 0, the convex hull of all selected points is generated.
All other values of alpha generate an alpha hull,
where faces are only constructed if all edge lengths are less than alpha.
This parameter thus provides support for generating target hulls with 
concave surfaces and/or disjoint polygons.
Note that when alpha is low, more distal points may be excluded from the hull.

In simple cases, the convex hull may be preferred for speed and simplicity. 
However,the alpha-hull is useful and sometimes necessary
in images where similar background colours are present.

#### Delta
The delta parameter determines the extent of the target-space outside the target [hull](#show-hull),
with higher values increasing the target colour volume.
For images where the target colours are very distinct from background, 
a simple hull combined with a high delta value is sufficient to identify the target pixels.
However, for more images with background colours closer to the target, lower delta is necessary and
more points will likely be needed to accurately define the target hull.

It is recommended to use a delta value of at least 1, at least for the final analysis,
to account for the variation introduced by [voxel](#voxel-size) down-sampling.
To improve responsiveness in the interactive panel, 
pixels mapped to voxels found inside or within delta of the hull are considered as target.
However, in the [area analysis](#area), each pixel is considered independently.

#### Fill
Local heterogeneity is a complex issue in image segmentation.
Frequently, a few isolated pixels may not fall within the target colour, despite being part of the target subject.
To account for this, the option to fill areas surround by target pixels is provided. 
The maximum size of filled contiguous regions (in pixels) should be entered into "fill".

#### Remove
As for [fill](#fill), a few isolated pixels may fall within the target colour volume, 
despite not being part of the target subject. 
To account for this, the option to remove pixels surrounded by non-target pixels is provided. 
The maximum size of removed contiguous regions (in pixels) should be entered into "remove".

#### Hull from mask
This button will only be active if an image mask has been [loaded](#load-mask).
When clicked, voxels mapping to pixels selected in the image mask,
representing at least [min pixels](#min-pixels) are first identified.
These points are then considered for hull construction using the [alpha](#alpha) parameter.
Vertices of this constructed hull are then added as [priors](#priors).

This is useful to rapidly generate a target full from existing image masks. 
Such a hull can then be adapted in AlGrow to improve robustness of the target hull across images.

#### Show Hull
The constructed target hull is displayed in the 3D plot. 
You can show/hide this hull by toggling this button.
You can also change the colour of the hull with the tool below.

#### Show Selected
Pixels corresponding to selected voxels are displayed in the image preview.
You can enable/disable this highlighting by toggling this button 
You can also change the colour of the highlighting with the tool below.

#### Show Target
Pixels corresponding to voxels that are determined to be 
inside or within [delta](#delta) of the target hull.
are displayed in the image preview.
You can enable/disable this highlighting by toggling this button 
You can also change the colour of the highlighting with the tool below.

#### Selected
This vertical menu bar contains buttons for selected voxels. 
Each button details the voxel position in Lab colourspace in text,
and is presented in the approximate corresponding colour. 

Two buttons are found at the top of this selection:
##### Clear
This button removes all selected voxels
##### Reduce
Clicking this button removes any voxels from the selection that are not vertices of the constructed hull.
This is particularly useful when saving a configuration,
when proceeding to analysis of multiple images,
or to speed up interactions in this interface.

Reduce can occasionally result in unexpected outcomes with [alpha](#alpha) hulls. 
This is due to the Delauney triangulation method used in hull construction, which, 
when performed on this new set of points, will result in a different set of triangles.
The edges of these triangles are filtered by length and presence on the boundary
to determine those to be used in construction of the alpha hull.

#### Priors
This vertical menu bar is similar to the adjacent [selected](#selected).
However, priors are stored across images, and any voxels found in selected are copied to here when loading a new image.
Most importantly, this allows training and evaluation of a target hull across multiple images.

It should be noted that priors are no longer stored as voxels mapped to pixels.  
Priors still contribute as points in hull construction and are presented in the 3D-plot as spheres. 
Priors can also be removed by shift-clicking on the sphere in the 3D plot
or by clicking on the corresponding button in the priors panel.

The [clear](#clear) and [reduce](#reduce) buttons have the same function as found in the [selected](#selected) menu,
except that they operate on this set of priors.

#### Dice coefficient
The Dice similarity coefficient (aka Sørensen–Dice index, F1 score) 
is calculated when a true-mask is [loaded](#load-mask) for the current loaded image.
It is defined as two times the area of the intersection of A and B, divided by the sum of the areas of A and B.
Internally, it is calculated as 2tp / ( 2tp + fp + fn ), where tp=true positive, fp = false positive, fn = false negative.

The dice-coefficient is often preferred over accuracy in image segmentation 
due to the inclusion of true negative results in calculating accuracy.
This can result in high accuracy values despite low true-positive rates, 
which is particularly important in comparing classification success 
when the ratio of target area to background area is small. 

## Analysis
### Area
Area calculation will use the current configuration settings.
At the top of this panel are buttons to add a directory of images, 
add a single file, remove a single file or remove all files.
The selected image file names are loaded into the box below, 
and parsed by the [filename regular expression](#filename-regex) to display block and time details.

#### Filename Regex
A regular expression (a.k.a. regex) is a pattern that matches a set of strings.
Sub-patterns in regex can be captured with parentheses "(" and ")".

AlGrow takes advantage of Python named groups in the pattern definition.
Named groups are prepended with "?P<name>", where the name can be a chosen group name.
Named groups are supported in AlGrow for:
  - year,
  - month,
  - day,
  - hour,
  - minute,
  - second, and
  - block

The default configuration file contains the following regex as an example:

```.*(?P<year>[0-9]{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12][0-9]|3[01])_(?P<hour>[01][0-9]|2[0-4])h(?P<minute>[0-5][0-9])m_(?P<block>[a-z|A-Z|0-9]*).*```

By way of explanation, we can break the start of this down sequentially:

  - ````.*```` means allow for any preceding characters
  - ```(?P<year>``` is the start of the capture group named year
  - ```[0-9]``` matches any single digit number from 0-9
  - ```{4}``` matches four of the preceding pattern, i.e. four numbers from 0-9
  - ```)``` closes the year capture group.
  - ```-``` then matches the "-" character.

Much of the rest of the pattern is consistent, it is just designed to specifically match corresponding capture groups.
To break down the end of this pattern for block recognition:

  - ```m_``` matches the preceding underscore ("m_") characters
  - ```(?P<block>``` starts the capture group named block
  - ```[a-z|A-Z|0-9]*``` matches a series of any length of any lowercase (a-z) or uppercase (A-Z) or single digit numbers (0-9)
  - ```).*``` closes the capture group but allows for any trailing characters 

Hopefully you can now interpret this pattern to say that we will parse a filename that looks like:
  - "2023-11-29_17h03m_block1" to mean:
    - year: 2023
    - month: 11
    - day: 29
    - hour: 17
    - minute: 3
    - block: block1
    
With at least year, month, day matched a time will be provided, with hour minute and second defaulting to 0.

The time and block details are used in subsequent [growth](#growth) analysis.
If they cannot be parsed from the filename, they will not be found in the output area file.
The time can be added manually to the area file in the format: "YYYY-MM-DD HH:mm:ss".
Similarly, a simple text string can be added to the block field in the output area.csv file for growth analysis.

#### Detect layout
When checked, the configured layout parameters will be used to [detect the layout](#detect-layout).
If not checked, the layout will not be detected, and segmentation will proceed for the whole image.

#### Fixed layout
A fixed layout can be loaded here from a layout.csv file, see [save fixed layout](#save-fixed-layout).
When loaded, the fixed layout overrides layout detection.

#### Processes
The number of processes to launch for parallel image processing.

#### Debugging images
There are three levels of debugging image production.
  - "DEBUG" will generate figures from many steps in the process:
    - The loaded RGB image
    - The image channels in Lab
    - The delta E to the selected circle colour
    - Circle detection and plate clustering dendrogram
    - Row/Column clustering within plates
    - Circles mask
    - Distance from the target hull
    - Mask processing, including fill and remove steps
    - Boolean mask
      - Can be subsequently [loaded](#load-mask)
    - Overlay:
      - Outlines of the defined target
      - Target circles
      - Indices for plates and circles 
  - "INFO" will generate just the boolean mask and overlay
  - "WARN" will currently generate no debugging images and is not recommended.

#### Output directory
Set this to a suitable destination for the figures and area file etc. to be written to.

#### Calculate
This launches the area analysis. Please be patient and wait for it to complete. 
You can expect a single process for a single image to take about 30 seconds if including layout detection.
With a fixed layout (or no layout) segmentation should take just a few seconds per image. 
 
### Growth
Relative growth rate (RGR) can be calculated as the log difference in area over a given span of time.
Assuming constant growth rates and unbiased error in our measurements, with more timepoints 
we can estimate RGR as the slope of a line of best fit to log transformed area over time. 
AlGrow can load an [area](#area) file, and a [samples map](#samples-map) to perform this analysis.

Pressing "calculate" will perform the analysis and prepare a summary "RGR.csv" file.
This describes the RGR estimated for each individual 
and the residual sum of squares (RSS) from the line fit. 
Any individual where the RSS is higher than the median RSS across all individuals
plus 1.5 times the interquartile range (IQR), is flagged as a "ModelFitOutlier". 

Plots of area over time are also generated for each group according to the [samples map](#samples-map),
with the line fit in log(area) transformed back to units of area ("group_plots/group.png").
The legend to these plots describes the block, unit and RGR for each replicate. 
ModelFitOutliers are presented with dashed-lines.

Box-plots are also prepared, grouped according to the [samples map](#samples-map),
both with ("RGR.png") and without ModelFitOutliers ("RGR_less_outliers.png"). 
A "RGR_mean.csv" file is also generated describing the mean RGR within each group, 
excluding individuals flagged as ModelFitOutliers. 

Note that the assumption of constant RGR is not always reasonable, 
particularly given the presence of diurnal variation in growth for most plant and algal species. 
AlGrow does not attempt to correct for "seasonality" or any other variation or biases in its analysis, 
and we encourage careful consideration in your own analyses from the output area.csv. 
Also note that no attempt to estimate a "block" effect is made when reporting the mean RGR,
despite our use of this term to identify distinct image sets. We expect that experimental designs will vary, 
and these analyses are provided as a convenience function for simple reporting contexts.

Nevertheless, provided the sampling of within day time points across days is balanced, 
the slope of the line of best fit through all timepoints is still likely to provide a good estimate of RGR.
Taking advantage of specifying a [start](#start-day) and [end](#end-day) time-point,
can be useful to avoid biases caused by incomplete measurements from the first or last day of an experiment. 

#### Samples Map
The samples map file can be prepared in spreadsheet software, such as Microsoft Excel or LibreOffice Calc.

Three columns are required, with headers:
  - Block
    - This should correspond to the [block](#filename-regex) output in the area file.
  - Unit
    - This should correspond to the unit ID in the area file 
  - Group
    - This is the sample description over which replicates will be grouped for analysis.

#### Start day
This value is relative to the first time point in the provided area file, in units of days.
Data from prior timepoints will be excluded in preparing the line of best fit.

#### End day
This value is relative to the first time point in the provided area file, in units of days.
Data from subsequent timepoints will be excluded in preparing the line of best fit.



### Additional arguments available during launch only.
#### Voxel size
The default value for voxel size is 1. 
This can be modified during launch with the --voxel_size argument.

Voxel size is the resolution of the grid used to down-sample pixels to voxels.

#### Downscale
The default value for downscaling is 1, i.e. no downscaling. 
This can be modified during launch with the --downscale argument.

An integer value provided to --downscale  will reduce the image size before analysis.
This can be useful to speed up the interface for larger images and is provided for testing
or target hull construction from very large images. 
Be warned, however, this isn't handled terribly well.
The scale and other pixel measurements including circle diameter etc. are not adjusted.
As such scale and layout specifications should share the same downscale value as subsequent analyses.





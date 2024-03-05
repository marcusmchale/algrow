# AlGrow
View in [Github Pages](https://marcusmchale.github.io/algrow/) to see the embedded demonstration video,
or directly on [GitHub](https://github.com/marcusmchale/algrow/) for simple markdown display.

- Alpha-hull colour boundary in segmentation of multiplexed plant and algal images for growth rate analysis.

Algrow is a software tool for automated image annotation, segmentation and analysis. 
It was developed by [Dr Marcus McHale](https://github.com/marcusmchale) 
to support macro-algal disc and plant growth phenotyping in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/) of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [University of Galway](https://www.universityofgalway.ie/).

## Install

The easiest way to use the software may be to download a compiled binary from the latest [release](https://github.com/marcusmchale/algrow/releases).

However, if you are familiar with the python environment you can also install from PyPi. 
```pip install algrow```
Then you should be able to launch from the console:
```algrow```

For advice on more complex installations, including building your own binaries see the [build notes](./build_notes.md)

## Instructions

The below video is an example of the use of AlGrow, though for more detil you may also benefit from reading the [guide](guide.md).

<video src="https://github.com/marcusmchale/algrow/releases/latest/download/tutorial.mov" controls="controls" style="max-width: 1080px;"></video>

For the command-line interface, AlGrow must be launched with a complete configuration, either supplied as arguments or in a configuration file.
A complete configuration requires both a configured hull and either; a complete layout specification, a fixed layout file or provide False to the --detect_layout argument.
e.g.
'''algrow -i images -o output -l False'''

## Features:
  - Improved deterministic model for colour-based image segmentation
    - Hulls (convex or alpha) flexibly define the target colour volume in a given 3D colour space
      - Fixed thresholds are implicitly cubic selections
      - Target colours and radius/k-means are implicitly spherical selections
    - Hulls can be intuitively trained by a user clicking on any currently unselected target regions.
      - Either in the original image OR in the 3D projection of Lab colourspace
      - Delta parameter generalises the model
      - The same hull is easily shared/trained across multiple images
    - Faster than many clustering or learning approaches 
      - GPU support through Open3D
    - Suitable for developing ground truth images OR analysing large datasets.
  - Automated annotation
    - Automated target region detection using surrounding circles (e.g. typical pots)
    - Relative indexation, supports:
      - Multiple target units within arrays (plates/trays)
      - Multiple arrays of units (plates/trays)
      - Movement, provided relative positions are maintained
      - Rotation, snaps plates/trays to nearest image axis alignment 
    - Optionally expand target area outside detected target circle (--circle_expansion)
    - Alternative fixed target layout, custom target layout, or whole image processing.
  - Statistical analysis
    - Growth rate determination from slope of linear model of log-transformed area over time.
    - Outlier identification by assessing model fit parameters (RSS)
  - Interfaces (GUI/CLI)
    - Graphical interface is provided for configuration and analysis in desktop environments ([guide](./guide.md)) 
    - Command line interface is provided for subsequent high-throughput analyses
  - Quality-control (3 levels)
    - DEBUG, many figures are generated, including; circle detection and clustering dendrograms for annotation.
    - INFO, just the image mask and a summary overlay with outlined target and annotated indices.
    - WARN, no debug images generated.


## The long story...

In our experiments, lamina discs of macroalgae are placed in an array under nylon mesh and arranged into cultivation tanks
where seawater is circulated and images are captured by RaspberryPi computers. Growth rates are determined from the 
change in visible surface area of each lamina disc, which is determined using image analysis techniques.

The [previous quantification method](<https://academic.oup.com/plphys/article/180/1/109/6117624>) relied on
thresholds in each dimension of a chosen colourspace for image segmentation. 
Although this strategy is widely used in plant phenotyping, it suffers in less controlled imaging environments where the
subject may not always be readily distinguished from background. For example, in our apparatus for Ulva phenotyping, 
microalgal growth occupies a similar colourspace to the Ulva subject. Similarly, in our apparatus for Palmaria phenotying,
leaching pigments can accumulate on the surface of nylon mesh making the distinction of these colours more difficult.
These colour gradients also result in poor performance for existing solutions like kmeans clustering (e.g. KmSeg),
due to poorly defined decision boundaries.

To allow user supervised definition of the target colour decision boundary,
we have developed AlGrow, with an interactive graphical user interface (GUI).
In this interface, the pixel colours are presented in a 3-dimensional (3D) plot in Lab colourspace. 
Colours can be selected by shift-clicking, either in this 3D plot or on the source image. 
When sufficient colours are selected (>=4), a 3D-hull can be generated, 
either the convex hull or an alpha hull which permits concave surfaces and disjoint regions. 

To automate annotation, we implemented a strategy to detect circular regions of high contrast.
This readily detects the subjects in our apparatus, 
due to high contrast holding rings (now using paint-marker applied to a different apparatus, yet to be reported)
but also the circular surface of typical plant pots against a contrasting background.
We then cluster these circles into plates/trays and assign indices based on relative positions. 
Importantly, this method of relative indexation supports movement of plates and trays across an image series, 
replacing a previously time-consuming process.

### Issues specific to prior strategy
Manual adjustment of fixed thresholds for segmentation in ImageJ is time-consuming and
can fail to identify or accurately segment across variable subject colours.
Manual curation can also introduce operator error and biases to area quantification.

Another key issue with the manual curation pipeline using ImageJ is the requirement to load all images in a stack into memory concurrently.
A number of pre-processing steps were employed to handle the scale of data from a single tank for the duration of a typical experiment (1 week) in a single processing step; 
  - file compression to jpeg format (lossy format resulting in deterioration of source data)
  - Image averaging
    - introduces error when there is movement due to binary thresholding,
    - blindly incorporates disturbed images, such as periods of lighting change or undetected operator intervention. 
  - pre-processing of images with wide thresholds to remove "known" background and reduce file size
    - Experimental conditions changed and the thresholds were no longer suitable but these thresholds were not always adapted.
    - In some cases raw images were discarded on the assumption that the hourly images were sufficient. 
      - these were sometimes inappropriately masked and so image analysis was no longer feasible.

### Our solution
The AlGrow application was developed to;
  - Independently process images from a time series according to a fixed, deterministic model of colour.
    - This reduces memory restrictions for large stacks of images seen with ImageJ
  - Rely on relative positions of image markers rather than absolute positions for image annotation
    - We use blue circles surrounding each position in the array, either included as plastic rings or painted on the surface of the frame.  
    - This removes the requirement for manual ROI definition and supports movement of plates/tanks/cameras between images.
    - Our automated annotation strategy also allows for flexible arrays and arrangements.
  - Support complex target colour spaces defined by alpha hulls
  - Provide an image debugging pipeline for parameter tuning and investigation of new or aberrant image sets.
  - Provide quality control outputs to allow inspection of data quality and improve calibration where needed.
  - Standardise and improve analysis by use of linear regression.
  - Identify experimental and or image segmentation issues by considering RSS of linear regression models
    - This allows us to reliably identifier outlier datapoints that do not reliably estimate lamina disc growth.


## Target area quantification method
    1. Optionally identify the target layout 
        1.1. A grayscale image is constructed reflecting the distance (delta-E) from the defined circle colour (skimage)
        1.2. Canny edge detection and Hough circle transform is applied to indentify target circles
        1.3. Circles are clustered to identify groups of defined size (plates) and remove artifact circles, e.g. reflections (Scipy.cluster.hierarchy)
        1.4. Orientation and arrangement of circles and plates is considered to assign indexed identities to each circle (Scipy.cluster.hierarchy)
        1.5. A layout mask is constructed to restrict further analysis to target areas

    2. Determine target subject area
        1.1 A boolean mask is determined by pixel colour being within the hull or within delta of its surface
        2.2 Small objects (--remove) are removed and filled (--fill) (skimage.morphology)
        2.3 Area of the mask within each circle is determined and output to a csv file.

    3. Analysis
        3.1 RGR is calculated as the slope of a linear fit in log area values (over defined period)
        3.2 Figures and reports are prepared

## To consider for future development
### UI
  - Change area_file/outdir handling to have distinct values for input/output
    - this may help user experience in the GUI, the current behaviour can be confusing.
  
### Features:
  - Analysis
    - fit to dynamic region, find area that best fits log-linear growth rather than using a fixed period
    - blocking (mixed effects models):
      - "block" and/or "plate"  
    - seasonal adjustment for diurnal variation when calculating RGR
      - eg. ARIMA x11
      - irrelevant when a single photo per day
      - if using complete days of data it should still be balanced, but the fit would improve outlier detection
      - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/stl
    - Remove outlier images before considering removing replicates

### Beyond the current scope
  - circle colour as hull rather than a fixed point
    - since we use a distance image and edge detection this isn't so important as it is in boolean thresholding
  - kmeans and other clustering methods (e.g. dbscan) to automate/simplify user input to hull definition.
    - kmeans example  https://www.sciencedirect.com/science/article/pii/S0167865519302806?via%3Dihub 
    - currently prefer supervised as kmeans performs poorly, also considered dbscan but not great
  - consider loading voxels across a whole series of images to improve visualisation across these.
  - consider higher dimensional space with e.g. texture features  
    - could include another 3d plot beside Lab. The same hyper-hull being represented across both plots? or separate hulls
    - maybe consider them as separate masks for target selection, taking the overlap.


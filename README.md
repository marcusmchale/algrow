
# AlGrow

- Alpha-shape defined colourspace for image segmentation of multiplexed image sequences

## The short story 

Algrow provides automated image analysis for measuring growth of macroalgal lamina discs. 
It was initially developed to suit the experimental apparatus in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/)
of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [University of Galway](https://www.universityofgalway.ie/). 
However, strategy is easily generalised to other contexts such as...
   - todo test with arabidopsis images and more

Features/parameters:
- Suitable for large sets of images 
  - Command line tool for ease of use on a server/cluster
  - Multiprocessing (-p num_cores)
- Quality-control overlays (option -q)
- Debugging pipeline (-D) for tuning parameters and examining each step of processing
- Automated annotation
  - Layout is determined by enclosing circles grouped into plates
    - Circle detection in chosen channel (-cc)
    - Search for circles expands criteria when insufficient are detected
    - circle expansion (-ce) to include area surrounding target circles (still must not overlap)
    - Pseudo-grid arrangement
      - Find clusters of circles of known size and arrangement (plates)
        - Number of circles per plate can be defined (-cpp)
      - Find arrangement of plates
        - Number of plates per image can be defined (-npi) 
      - Arrangements are simple sequences within rows (default) or columns for circles within plates (-ccf) or plates within images (-pcf)
        - ID incrementation direction is customisable (-clr, -ctb, -plr, -ptb)
- 
- Segmentation used for calibration GUI 
  - SLIC algorithm 
    - Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, TPAMI, May 2012. DOI:10.1109/TPAMI.2012.120
    - https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO
    - Irving, Benjamin. “maskSLIC: regional superpixel generation with application to local pathology characterisation in medical images.”, 2016, arXiv:1606.09518

- Interactive definition of alpha shape hull that specifies target colourspace.

- Noise removal
  - Fill small holes
  - Remove small objects 

- Analysis
  - Linear regression of log transformed values determines daily relative daily growth rate (RGR)
  - Data exported as csv files, including raw values, RGR and per group summaries
    - Summaries provided with and without filtering for outliers 
      - Outliers identified by residual sum of squares (RSS) from disc growth RGR model (> 1.5 * IQR)
  - Plots exported as png images (area estimates over time for each disc, model fits and box-plots of RGR)
  - Requires:
    - ID file (csv) containing the "block", "well" and "strain" supports automated analysis (-id).
    - regex patterns defined in config file to pull out block number and date/time from image filenames (-tr, -tf, -br).
- Quality control
  - debugging pipeline with images for each step and dendrograms for clustering in layout detection (-D)
  - overlay images with the final image mask, defined circles and plate/circle identifiers (-q) 

reading: # https://www.sciencedirect.com/science/article/pii/S0167865519302806?via%3Dihub
the above uses kmeans for initial clustering, could be doing this but prefer supervised for a single target cluster

## Get started
apt install python3.10
apt install python3.10-venv


### Distribution

  - Download 
    - get the latest [dist](https://github.com/marcusmchale/algrow/dist)
  - Install
    - pip install algrow.whl
  - Run
    - cd path/to/algrow/sample_images
    - mkdir raw
    - tar -xvzf sample_images.tgz -C raw
    - algrow.py -p 4 -q -i raw -id id.csv -o . -l info > log.txt

# this works on python 3.9 
# i think it also may need python3-tk on system?, 
# something strange to get the lasso selector working anyway
# 


### Run from source

  - Download
    - git clone https://github.com/marcusmchale/algrow
  - Set up virtual environment (recommended)
    - python3 -m venv venv
    - source ./venv/bin/activate
  - Run
    - algrow.py -i sample_images


### Build

  - Download
    - git clone https://github.com/marcusmchale/algrow
  -  Install python
    - sudo apt install python3.10 python3.10-venv python3.10-distutils python3.10-dev
  - Install dependencies for wxpython build
    - sudo apt install dpkg-dev build-essential python3-dev freeglut3 libgl1-mesa-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev libjpeg-dev libnotify-dev libpng-dev libsdl2-dev libsm-dev libtiff-dev libwebkit2gtk-4.0-dev libxtst-dev
  - Set up virtual environment (recommended)
    - python3.10 -m venv venv
    - source ./venv/bin/activate
    - pip install --upgrade pip
    - pip install dist/algrow-0.3-py3-none-any.whl
  - Build
    - python3 -m build



## The long story...

In our experiments, lamina discs of macroalgae are placed in an array under nylon mesh and arranged into cultivation tanks
where seawater is circulated and images are captured by RaspberryPi computers. Growth rates are determined from the 
change in visible surface area of each lamina disc, which is determined using image analysis techniques.

The [previous quantification method](<https://academic.oup.com/plphys/article/180/1/109/6117624>) relied on
simple thresholds in each dimension of a chosen colourspace for image thresholding. 
Although this strategy is widely used in plant phenotyping, it suffers in less controlled imaging environments where the
subject may always be easily distinguished from background. For example, in our apparatus for Ulva phenotyping, 
microalgal growth occupies a similar colourspace to the Ulva subject. Similarly, in our apparatus for Palmaria phenotying,
leaching pigments can accumulate on the surface of nylon mesh making the distinction of these colours more difficult.

Importantly however, the colours are distinct, and the issue is a matter of more appropriately defining the colourspace. 
We have developed a methodology, whereby a concave polygon is used to more accurately describe the target colourspace.
To define this polygon we construct an alpha shapes from a set of training points and implemented an interactive configuration tool for this purpose..

#todo Provide examples of convex/concave polygons in image segmentation.

Further, the prior methodology relied on manual annotation in the ImageJ graphical interface.
This process is time-consuming and dependent on fixed coordinate positions of the lamina disc across the image stack.

# Issues specific to existing strategy
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
    - In some cases raw images were discarded on the assumption that the hourly images were sufficient. But these were sometimes inappropriately masked and so image analysis was no longer feasible.

# Our solution
The AlGrow application was developed to;
  - Independently process each image
    - This resolves the issues of memory restrictions for large stacks of images
  - Rely on relative positions of image markers rather than absolute positions for image annotation
    - We use blue circles surrounding each position in the array, either included as plastic rings or painted on the surface of the frame.  
    - This removes the requirement for manual ROI definition and supports movement of plates/tanks/cameras between images.
    - Our automated annotation strategy also allows for flexible arrays and arrangements.
  - Support complex target colour spaces defined by alpha hulls
  - Provide an image debugging pipeline for parameter tuning and investigation of new or aberrant image sets.
  - Provide quality control outputs to allow inspection of data quality.
  - Standardise and improve analysis by use of linear regression models rather than simple averaging across timepoints.
  - Identify experimental and or image segmentation issues by considering RSS of linear regression models
    - This allows us to reliably identifier outlier datapoints that do not reliably estimate lamina disc area.


# Instructions
## Calibration
The calibration GUI will launch unless scale, circle colour and hull vertices are provided
as arguments or in a configuration file. There are utilities provided to define each of these:
### Set scale
An internal reference with known dimensions should be included in the image. 
This tool allows you to use this internal reference to calculate an appropriate scale
to convert from pixels to units of physical distance and area (mm²).

Click once to start drawing a line and again to finish the line. This will provide a value in pixels (px)
to the toolbar below. Type the physical distance (mm) for this line into the next box and press enter. 
This will calculate a scale (px/mm). 

It may help to zoom in on the image to accurately define the start and end of the line. 
Use the navigation toolbar above to zoom and pan on the image. 

Alternatively you can enter the scale manually into the box and press enter.
Click "save and close" to accept this scale, or click "clear" to start again.
### Circle colour
To detect the layout of lamina discs, AlGrow requires circles surrounding each subject in a pseudo-grid layout. 
These circles must be of a uniform colour, ideally one that is readily distinguished from the target subjects.

To define this colour, a picture is loaded and an interactive "lasso" allows you to encircle part of the image 
that contains includes this colour. As for defining the scale, you may wish to zoom in on part of the image to do so.
Multiple areas can be selected and the median colour of these pixels will be chosen. 

The selected colour is displayed in the box beside the save and close button. 
You can always "clear" the selection and start again or "save and close" to accept the value and continue
### Target hull
This step takes advantage of layout detection,so it is not available if circle colour is not defined.
It will also fail if the layout parameters are poorly defined.

Launching this window will take some time so please be patient.
The layout will be detected for a sampled set of images will be segmented (SLIC).
Images are processed in parallel so if you have specified to use a large number of images for calibration (--num_calibration) 
you might also consider using a similar number of processes (--processes) to speed things up.

Two views are presented. 
On the left, the image and segment boundary overlay allows you to click to select/deselect
target segments.  Click here to select target segments, they will be highlighted (blue).

On the right, the median segment colours are plotted in Lab colourspace. 
You can click and drag on this image to change the perspective and better visualise the separation between colours.

When enough segments are selected from the image on the left (>=4 while alpha = 0)
a hull will be plotted on the right. 
Segments whose median colour is contained within this hull, 
or within delta of the surface will be highlighted (green) on the image on the left.

You can turn on/off segment highlighting by clicking the blue/green buttons
corresponding to the "selection" and "within" colours. It is useful to start with a higher delta value (e.g. 10) 
then reduce this parameter slowly to identify points that are just outside the hull. 
The objective is to efficiently select the boundary points defining our target colourspace (our hull vertices).
Points that do not form a vertex of this hull will be discarded after calibration.

You can navigate between images by clicking the prev/next buttons on the bottom toolbar. 
When you are satisfied with the defined hull click save and close. 
This also may take some time, particularly if the --animate option is specified and a complex hull is defined.
A debugging image is created depicting the points from the set of sampled images in Lab colourspace
with the defined hull overlayed.

#### Advanced
The alpha parameter is necessary to construct a concave hull, 
which may be necessary when similar background colours are present in the image.
When alpha is 0, the convex hull is constructed. 
When alpha is too high, some points will be excluded.
For a given set of selected points you can "optimise" the alpha parameter by clicking the "A" button.

If SLIC segmentation is not perfect, i.e. some segments contain both subject and background, do not worry. 
Just be sure to select segments that only contain your target. 
However, you may choose to optimise the SLIC parameters to better suit your images.

### Continue
When all of these are complete (green) you will be able to click "continue".
This will write out file specifying the configuration parameters into the output file. 
You can copy this into your configuration file to avoid repeating configuration
with similar images and to ensure consistency across multiple analyses. 


## Target area quantification method
    1. Identify layout 
        1.1. A grayscale image is constructed reflecting the distance (delta-E) from the defined circle colour (skimage)
        1.2. Canny edge detection and Hough circle transform is applied to indentify target circles
        1.3. Circles are clustered to identify groups of defined size (plates) and remove artifact circles, e.g. reflections (Scipy.cluster.hierarchy)
        1.4. Orientation and arrangement of circles and plates is considered to assign indexed identities to each circle (Scipy.cluster.hierarchy)
        1.5. A layout mask is constructed to restrict further analysis to target areas

    2. Determine subject area
        1.1 A boolean mask is determined by pixel colour being with alpha hull or within delta of its surface
        2.2 Small objects (--remove) are removed and filled (--fill) (skimage.morphology)
        2.3 Area of the mask within each circle is determined and output to a csv file.

    3. Analysis
        3.1 RGR is calculated as the slope of a linear fit in log area values (over defined period)
        3.2 Rigures and reports are prepared


# todo
  - alternatives to distance calculation, the "contains" approach is far faster
    - consider methods to buffer the hull by delta and use points of this new hull

  
# To consider
  - multiple target circle colours (easy to implement with distance calculation)
  - GUI window for date, time, block regex from filename in calibration
  - Process image filename during loading to provide date time block etc. rather than at the final analysis
    - not necessary until writing out but might be useful for annotation in debugging loaded image.
  - Alternative/supplementary layout detection methods
  - Support superpixel segmentation during area calculation (as was done in earlier versions)
    - tradeoff: the time added for slic might be regained in the hull distance calculation
    - without fill/remove/blurring this may be necessary/improve results 
    - consideration: sometimes superpixel segmentation performs poorly, e.g. Palmaria
  - Downscaling images to speed up colour calibration, in particular the hull segmentation etc.
  - Analysis (maybe overkill)
    - fit to dynamic region, find area that best fits log-linear growth rather than using a fixed period
    - blocking (mixed effects models):
      - "block" and/or "plate"  
    - consider seasonal adjustment for diurnal variation: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/stl


# To explore
 - consider how the alpha parameter can allow for multiple disconnected regions
   - assumes similar density in clouds but this is expected from SLIC
   - could be handy for subjects with mixed colours for tissues, e.g. branches and leaves.
 - consider higher dimensional space e.g. texture features  
   - could include another 3d plot beside Lab. The same hyper-hull being represented across both plots? or separate hulls
   - maybe consider them as separate spaces/masks for target selection?


# Beyond the current scope but maybe one day
  - Consider developing image capture using libcamera2 and apscheduler - run as daemon
    - Consider not compressing to jpg
    - provide HDR (even with pi2 camera) by capturing a series of images at multiple exposures and compositing
  - Night vision:
    - Motorised IR cut camera (supplier pimoroni is on agresso but not willing to supply)
      - https://www.uctronics.com/arducam-noir-8mp-sony-imx219-camera-module-with-motorized-ir-cut-filter-m12-mount-ls1820-lens-for-raspberry-pi.html
    - Note: IR didn't work well in testing with a fixed camera - very high background reflection from the water surface and plates/frames

# Comparisons with other tools/frameworks 
## plantcv2
  - Requires customised code
  - Now has auto-grid detection...https://www.authorea.com/users/508818/articles/591710-simplifying-plantcv-workflows-with-multiple-objects
    - Relies on a simple grid layout (we support a nested structure in rows or columns),
    - Does not require markers, handles some missing points
      - we could work towards a similar layout detection, not super hard to implement, but still need to provide some parameters

  - A variety of thresholding methods, but nothing like the alpha hull


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5895192/


# TO reference:
 - colour volume as alpha shape in Methods in Ecology and Evolution 2020  https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13398

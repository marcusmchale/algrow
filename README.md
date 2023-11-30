
# AlGrow

- Alpha-hull colour boundary in segmentation of multiplexed plant and algal images for growth rate analysis.

## The short story 

Algrow is a software tool for automated image annotation, segmentation and analysis. 
It was initially developed to support macro-algal disc phenotyping in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/)
of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [University of Galway](https://www.universityofgalway.ie/), and has since been applied to other images, such as arabidopsis.


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
thresholds in each dimension of a chosen colourspace for image segmentation. 
Although this strategy is widely used in plant phenotyping, it suffers in less controlled imaging environments where the
subject may not always be readily distinguished from background. For example, in our apparatus for Ulva phenotyping, 
microalgal growth occupies a similar colourspace to the Ulva subject. Similarly, in our apparatus for Palmaria phenotying,
leaching pigments can accumulate on the surface of nylon mesh making the distinction of these colours more difficult.
These colour gradients also result in poor performance for existing solutions like kmeans clustering (e.g. KmSeg).

To more allow user supervised definition of the target colourspace, we have developed in AlGrow, an interactive GUI.
In this interface, the pixel colours are presented in a 3-dimensional (3D) plot in Lab colourspace. 
Colours can be selected either in this 3D plot or from the source image. When sufficient colours are selected (>=4), 
a hull can be generated, either the convex hull or an alpha hull which permits concave surfaces and disjoint regions. 

To automate annotation, we implemented a strategy to detect circular regions of high contrast.
This readily detects the subjects in our apparatus, due to high contrast holding rings (now paint-marker applied to the apparatus)
but also the circular surface of typical plant pots.
We then cluster these circles into plates/trays and assign indices based on relative positions. 
Importantly, this method of relative indexation supports movement of plates and trays across an image series, 
easing the previously time-consuming process.

# Issues specific to prior strategy
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

# Our solution
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
        1.1 A boolean mask is determined by pixel colour being within the alpha hull or within delta of its surface
        2.2 Small objects (--remove) are removed and filled (--fill) (skimage.morphology)
        2.3 Area of the mask within each circle is determined and output to a csv file.

    3. Analysis
        3.1 RGR is calculated as the slope of a linear fit in log area values (over defined period)
        3.2 Figures and reports are prepared

# To consider for future feature development
  - circle colour as hull rather than a fixed point
  - kmeans and other clustering methods to automate/simplify user input to hull definition.
  - Analysis 
    - fit to dynamic region, find area that best fits log-linear growth rather than using a fixed period
    - blocking (mixed effects models):
      - "block" and/or "plate"  
    - seasonal adjustment for diurnal variation when calculating RGR
      - eg. ARIMA x11
      - irrelevant when a single photo per day
      - if using complete days of data it should still be balanced, but the fit would improve outlier detection
      - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/stl
    - Remove outlier images before considering removing whole replicates

# Beyond the current scope but maybe one day
 - consider loading voxels across a whole series of images to improve visualisation across these.
 - consider higher dimensional space with e.g. texture features  
   - could include another 3d plot beside Lab. The same hyper-hull being represented across both plots? or separate hulls
   - maybe consider them as separate masks for target selection, taking the overlap.

# TODO
  - write documentation
  - package as installer
  - Evaluate/test
    - use kmseg to generate ground truth images and evaluate classification accuracy, naiive kmseg vs corrected with algrow. 
    - test across images with the trained hull. c.f. ML. classifiers.
    - test interaction between true-mask and layout-mask

# Note 
-  that these interactive representations of pixel colour can improve:
  - understanding of colour representations that can then be feed back into design of imaging apparatus
    - e.g. overlapping clusters 
 - AlGrow is successful where km-seg fails to accurately separate clusters
   - i.e. a decision boundary is required that is not reliably determined by k-means clustering
   - both rely on supervision
   - Algrow also works across images without need to train another model.  
  
 - Algrow did not use other colour representations (RGB, HSV) as Lab benefits from mostly uniform distance (delta E) 
    - However the alpha hull concept would still be useful in other colourspaces 
    - We did not use more advanced delta E calculations, favouring the simplicity of euclidean distance.
 
  - the reduce function (removing points not in the hull) with alphashape
    - the Delaunay triangulation is then modified! 
    - This means the alpha hull is then selected from a different set of triangles
    - this frequently results in non-watertight structures 
      raising the alpha can help to recover from this.
  - the alpha hull allows for multiple disconnected regions
   - handy for subjects with mixed colours for tissues, e.g. stems, leaves, flowers, diseased/stressed/exposed tissues.
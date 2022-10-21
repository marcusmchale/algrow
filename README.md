
# DiscGrow

- Automated multiplexed area quantification for growth rate determination of macroalgal lamina discs

## The short story 

Discgrow provides automated image analysis for measuring growth of macroalgal lamina discs. 
It was developed to suit the experimental apparatus in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/)
of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [National University of Ireland, Galway](https://www.nuigalway.ie/). 
However, parameters are provided that should allow its application in other contexts 

Features:
- Multiprocessing (-p num_cores)
- Quality-control overlays (option -q)
- Debugging pipeline (-D)
- Annotation
  - Layout is determined by enclosing circles grouped into plates
    - Circle detection in chosen channel (-cc)
    - Search for optimal accumulator threshold (param2 in HoughCircles) or fixed to improve performance (-ph)
    - Pseudo-grid arrangement (i.e. sequential rows or columns of circles and plates)
      - Number of plates and circles per plate can be defined (options -npi and -cpp)
      - ID incrementation direction is customisable (-crf, -clr, -ctb, -prf, -plr, -ptb)
- Segmentation
  - SLIC algorithm
    - R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 11, pp. 2274-2282, Nov. 2012, doi: 10.1109/TPAMI.2012.120.
  - optional interactive selection of representative regions for target colour selection
    - run only if the colour (Lab colour-space) is not defined (-tL, -ta, -tb)
  - Area is reported for closest cluster to the defined target 
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

## Get started
### Distribution

  - Download 
    - get the latest [dist](https://github.com/marcusmchale/discgrow/dist)
  - Install
    - run pip install discgrow.whl
  - Run
    - cd path/to/discgrow/sample_images
    - mkdir raw
    - tar -xvzf sample_images.tgz -C raw
    - discgrow.py -p 4 -q -i raw -id id.csv -o . -l info > log.txt

### Run from source

  - Download
    - git clone https://github.com/marcusmchale/discgrow
  - Set up virtual environment (recommended)
    - python3 -m venv venv
    - source ./venv/bin/activate
  - Run
    - discgrow.py -i sample_images


### Build

  - Download
    - git clone https://github.com/marcusmchale/discgrow
  - Set up virtual environment (recommended)
    - python3 -m venv venv
    - source ./venv/bin/activate
  - Build
    - python3 -m build



## The long story...

In our experiments, lamina discs of macroalgae are placed into 6 well plates under nylon mesh and
surrounded by a blue holding ring. These plates are arranged into cultivation tanks
where seawater is circulated and images are captured by RaspberryPi computers and cameras positioned over each tank.
The area of each lamina disc can then be calculated over time using image analysis techniques
to determine growth rates for each macroalgal strain.

The [previous quantification method](<https://academic.oup.com/plphys/article/180/1/109/6117624>)
relied on annotation by sequential selection of the well region from a stack of images in the ImageJ graphical interface.
The manual annotation process is time-consuming and dependent on fixed coordinate positions
of the lamina disc across the image stack.
Manual adjustment of fixed thresholds for segmentation in ImageJ are also time-consuming and
can fail to identify or accurately segment atypical images.
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

The DiscGrow application was developed to;
  - independently process each image to resolve the issues of memory restrictions for large stacks of images
  - relying on relative rather than absolute positions to remove the requirement for ROI definition and handle movement of plates/tanks/cameras between images.
  - remove/reduce operator input into segmentation by using cluster based methods rather than thresholding.
  - provide an image debugging pipeline for investigation of new or aberrant image sets.
  - provide quality control pipelines to allow manual investigation of data quality and tune parameters to new image sets.
  - automate annotation for flexible plate types, numbers and arrangements. 
  - standardise and improve analysis by use of linear regression models rather than simple averaging across timepoints
  - identify outliers by considering RSS of linear regression models to highlight possible experimental issues.


Steps:

    1. Circles are identified by Hough Circle transform in median blur of selected channel of Lab colourspace (OpenCV)
    2. Clusters of circles are identified to define plates and remove artifact circles (Scipy.cluster.hierarchy)
    3. Orientation and arrangement of circles and plates are determined to define well identities (Scipy.cluster.hierarchy)
    4. SLIC segmentation in Lab colourspace to label target regions (skimage.segmentation)
    5. Remove small objects and fill small holes (skimage.morphology)
    5. The area of the image mask is determined within each annotated circle and written to a csv file.
    6. RGR analysis is performed, figures and reports are generated.

# Todo
  - Segmentation
    - Replace opencv with skimage, should be able to do all this in the one library
    - Generate labeled mask circles on first iteration through plates
      - use this to speed up counting pixels
  - Data management (probably unnecessary)
    - Service node and server design, each pi as a node and a central server to gather and present reports.
      - Capture images directly and use raw data rather than jpg
        - Consider rolling shutter in raspberry Pi cameras, is it really an issue?
          - Could average multiple images as done before
  - Analysis (to consider but maybe overkill)
    - fit to dynamic region, find area that best fits log-linear growth rather than using a fixed period
    - blocking (mixed effects models):
      - "block" and/or "plate"  

# Comparisons with other tools/frameworks 
## plantcv2
  - Command line application does not require coding in Python
  - Multiplexing does not rely on simple grid layout (we support a nested structure in rows or columns)
  - Relative rather than absolute positioning 
    - Allows movement of camera or subjects
    - Disadvantage is we require markers being distinguishable in the image (e.g. blue circles)

    

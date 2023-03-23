
# DiscGrow

- Graph-based detection of target superpixels for automated growth rate determination in multiplexed images of macroalgal discs

## The short story 

Discgrow provides automated image analysis for measuring growth of macroalgal lamina discs. 
It was developed to suit the experimental apparatus in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/)
of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [University of Galway](https://www.universityofgalway.ie/). 
However, parameters provided allow it to be applied in other contexts.

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
- Segmentation
  - SLIC algorithm 
    - Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, TPAMI, May 2012. DOI:10.1109/TPAMI.2012.120
    - https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO
    - Irving, Benjamin. “maskSLIC: regional superpixel generation with application to local pathology characterisation in medical images.”, 2016, arXiv:1606.09518
- Graph based selection of target
  - Interactive selection of representative regions for selection of target colour(s)
  - Nodes within colour distance (deltaE, --target_dist) of target colour are selected
  - Adjacent nodes within colour distance (--graph_dist) are also selected
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

# note so far this works on python 3.9 
# i think it also may need python3-tk also installed for interactive colour picker


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

The DiscGrow application was developed to;
  - independently process each image to resolve the issues of memory restrictions for large stacks of images
  - rely on relative rather than absolute positions to remove the requirement for ROI definition and handle movement of plates/tanks/cameras between images.
  - remove/reduce operator input into segmentation by using cluster based methods rather than fixed thresholds.
  - support mixed target colours
  - provide an image debugging pipeline for parameter tuning and investigation of new or aberrant image sets.
  - provide quality control pipelines to allow manual investigation of data quality.
  - automate annotation for flexible plate types, numbers and arrangements. 
  - standardise and improve analysis by use of linear regression models rather than simple averaging across timepoints
  - identify outliers by considering RSS of linear regression models to highlight possible experimental issues.


Simple steps:

    1. Selection of target colour regions in interactive window (--target_colour).
    2. Identify target circles by canny edge detection and Hough circle transform in selected channel of Lab colourspace (skimage)
    3. Cluster circles to define plates and remove artifact circles (Scipy.cluster.hierarchy)
    4. Determine orientation and arrangement of circles and plates to relate each target to its indexed identity (Scipy.cluster.hierarchy)
    5. SLIC segmentation in Lab colourspace to identify superpixels within circles (skimage.segmentation)
    6. Graph construction with superpixels as nodes and colour-distance weighted edges to adjacent nodes (networkx, skimage.future, skimage.colour.deltaE_ciede2000)
    7. Selection of initial target nodes below a threshold distance from the selected set of target colours (--target_dist)
    8. Extension of target nodes, expanding from initial target nodes to include connected superpixels below a threshold colour distance (--graph_dist)
    9. Create mask from selected target pixels, remove small objects and fill small holes (skimage.morphology)
    10. Calculate area of the mask within each circle and write to a csv file.
    11. Perform analysis (calculate RGR over defined period) and prepare figures and reports.


Tuning:

The first step is run when no --target_colour values are provided as arguments or in the configuration files.
We advise to perform colour selection first with a single representative image as input and copy the output into a configuration file to run across a time-series.
With highly variable target colours it may be useful to run the colour-selection on multiple images and collect the outputs into a single list of target colours to use across the whole dataset.

Beyond target colours there are two key parameters that may require tuning, --target_dist and graph-dist. 
In general, you should raise these values until non-target pixels are detected, but the debugging pipeline can help to select an appropriate value. 

In step 7 above, the distance between the mean colour of each superpixel node and all target colours is calculated .
Those nodes where the minimum distance is below --target_dist will be considered as initial target nodes. 
To optimise this value you may want to consider the plot of "distance from targets" in the debugging pipeline.

In step 8 above, edges above --graph_dist colour-space distances are removed and nodes still connected to the initial target regions are included in the target area.
To optimise this value you may want to consider the representations of the full, truncated and background-removed graphs in the debugging pipeline.

# Todo
  - Make the graph step optional, multi-target superpixel distance may be sufficient in some cases
  - Segmentation
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
  - Interactive selection of colour doesn't work when run on server using X forwarding.

# Comparisons with other tools/frameworks 
## plantcv2
  - Command line application does not require customised coding
  - Multiplexing does not rely on a simple grid layout (we support a nested structure in rows or columns)
  - Relative rather than absolute positioning 
    - Allows movement of camera or subjects
    - Disadvantage is we require ring markers around each subject being distinguishable in the image (e.g. blue circles)
  - Multiple target colours and target colour distance rather than fixed thresholds
    - graph based expansion of search area to include similar adjacent colours

    

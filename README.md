
# DiscGrow

- Automated multiplexed area quantification for growth rate determination of macroalgal lamina discs

## The short story 

Discgrow provides image analysis for measuring growth of macroalgal lamina discs. 
It is currently optimised for the experimental apparatus in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/)
of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [National University of Ireland, Galway](https://www.nuigalway.ie/). 
However, with minor modifications to clustering and segmentation logic
it can be easily adapted to other experimental contexts.

Features:

    - Segmentation
        - Multiprocessing (-p num_cores)
        - Search for optimal accumulator threshold (param2 in HoughCircles) for circle detection 
          - can be fixed to a constant to improve performance (option -ph)
        - Quality-control overlays (option -q)
        - Debugging pipeline to adapt thresholds (-D)
    - Annotation
      - Layout is determined by circles clustered into plates
          - assumes as pseudo-grid like arrangement
              - i.e. sequential rows or columns of plates, with each plate aggregating multiple circles
          - Number of plates and circles per plate can be defined (options -npi and -cpp)
          - ID incrementation is fully customisable by command line arguments (-crf, -clr, -ctb, -prf, -plr, -ptb)
    - Analysis
      - Requires:
        - ID file (csv) containing the "tank", "well" and "strain" supports automated analysis.
        - regex patterns defined in config file to pull out "tank" number and date/time from image filenames.
      - Regression model of log transformed values to determine daily relative daily growth rate (RGR)
      - Data exported as csv files, including raw values, RGR and per strain summaries
          - Summaries provided with and without filtering for outliers 
            - Outliers identified by residual sum of squares (RSS) from disc growth RGR model (> 1.5 * IQR)
      - Plots exported as png images (area estimates over time for each disc, model fits and boxplots of RGR)


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

[Previous quantification methods](<https://academic.oup.com/plphys/article/180/1/109/6117624>)
rely on annotation by sequential selection of the well region from a stack of images in the ImageJ graphical interface.
The manual annotation process is time-consuming and dependent on the coordinate position
of the lamina disc being consistent throughout the image stack.
Manual procedures for segmentation and area determination in ImageJ are also time-consuming and
limited to simple approaches which can fail to identify or accurately segment atypical images.
The interactive strategy also suffers from a typical issue with manual curation,
the introduction of operator error and biases to area quantification.

Another key issue with the manual curation pipeline using ImageJ is the requirement to load all images in a stack into memory concurrently.
A number of pre-processing steps were employed to handle the scale of data from a single tank for the duration of a typical experiment (1 week) in a single processing step; 
  - file compression to jpeg format (lossy format resulting in deterioration of source data and segmentation quality)
  - Image averaging 
    - introduces error when there is movement due to binary thresholding,
    - blindly incorporates disturbed images, such as periods of lighting change or undetected operator intervention. 
  - pre-processing of images with wide thresholds to remove "known" background and reduce file size
    - Experimental conditions changed and the thresholds were no longer suitable but these thresholds were not always adapted.
    - In some cases raw images were discarded on the assumption that the hourly images were sufficient. But these were sometimes inappropriately masked and so image analysis was no longer feasible.

The DiscGrow application was developed to;
  - independently process each image
    - Relying on relative positions of blue rings surrounding each disk rather than absolute positions 
    - allows movement of plates/tanks/cameras between images,
    - and removes the limit that batch processing imposes on the frequency and duration of measurements.
  - remove/reduce operator input into thresholding
    - by providing pre-defined defaults for segmentation thresholds.
    - An image debugging pipeline is also provided identify optimum values under changed conditions.
  - provide quality control measures
    - including overlay images with the final image mask, defined circles and plate/circle identifiers. 
      - These can be used to verify the quality of the applied segmentation 
      - and inspect lamina discs that are highlighted by low quality regression models.
  - automate annotation
    - for the existing plates, plate layout and ID determination. 
    - Customising the "sort_plates" function in the Layout class would support adaptation to other conformations. 
  - standardise and improve analysis
    - using all of the available data.
    - We now fit linear regression models on log transformed area values over time,
    - to extract RGR as the slope of linear model.
    - RSS of these models can help to identify poorly fitting models, an indication of experimental issues.

Multiple images can be concurrently assessed (multiprocessing).
This application requires less resources than the alternative pre-processing steps (to be quantified) 
and can be performed on the Raspberry Pi directly (to be verified and timed).
Many atypical images are readily detected during circle and plate detection, 
and the availability of data for each image (rather than an aggregate images) 
improves detection of other anomalies and provides opportunity to account for outliers.  

Steps:

    1. Circles are identified by Hough Circle transform in median blur of b channel of Lab colourspace (OpenCV)
    2. Clusters of circles are identified to define plates and remove artifact circles (Scipy.cluster.hierarchy)
    3. Orientation and arrangement of circles and plates are determined to define well identities (Scipy.cluster.hierarchy)
    4. SLIC segmentation in Lab colourspace to label "green" regions (scikimage.morphology)
    5. Noise removal.
        a. Saturation (S) threshold to select high values, colour rich regions
        b. Value (V) threshold to identify dark regions that occur from folding of lamina tissues
        c. Join S AND V to create a colour mask
        d. Green-Red (a)  threshold to select green lamina
        e. Blue-Yellow (b) threshold to select green lamina
        f. Join a AND b to create green mask
        g. Join colour AND green to create the image mask
        h. Apply a circle mask around the perimeter of each well (from 2)
        i. remove small objects (skimage.morphology)
        j. fill small holes (skimage.morphology)
    5. The area of the image mask is determined within each annotated circle and written to a csv file.
    6. RGR analysis is performed (, figures and reports are generated.

# Todo
  - Layout 
    - support using in any grayscale image in circle detection
      - currently just using the "b" channel
  - Workflow changes to image segmentation, split into identification of:
          - target/not-target
          - alive/diseased/dead 
    - each process should be described in configuration file or similar.
    - Have a configuration file for each thresholding process to identify:
    - Provide options for handling low exposure elements (currently being included as target)
    - Use these to build the masks for quantitation
    
  - Data management
    - Service node and server design, each pi as a node and a central server to gather and present reports.
      - Capture images directly and use raw data rather than jpg
        - Consider rolling shutter in raspberry Pi cameras, is it really an issue?
          - Could average multiple images as done before
          
  - Analysis
    - Options to consider:
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

    

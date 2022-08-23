
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

    - Multiprocessing (-p num_cores)
    - Search for optimal accumulator threshold (param2 in HoughCircles) for circle detection 
        - can be fixed to a constant to improve performance (option -ph)
    - Quality-control overlays (option -q)
    - Debugging pipeline to adapt thresholds (-D)

## Get started
### Distribution

  - Download 
    - get the latest [dist](https://github.com/marcusmchale/discgrow/dist)
  - Install
    - run pip install discgrow.whl
  - Run
    - Disc area determination
      - basic
        - discgrow.py -i sample_images/
      - multi-threading (-p num_cores) with overlay output (-q) and progress information (-l info)
        - discgrow.py -i sample_images/ -p 4 -q -l info
      - image debugging (-D plot, requires matplotlib) for a single image and additional debugging details (-l debug)
        - discgrow.py -i sample_images/a.jpg -D plot -l debug
    - Growth rate analysis
      - discgrow.py -id sample_data/id.csv -o sample_data/
    - Both together
      - discgrow.py -s ./images -p 4 -id ./id.csv
    

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
    - python3 build



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
The existing strategy also suffered from a typical issue with manual curation,
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
  - provide independent processing of each image
    - support for limited movement of subjects between images
    - remove the limit that batch processing imposed on the frequency and duration of measurements
  - remove/reduce operator input into thresholding
    - pre-defined defaults for segmentation thresholds with an image debugging pipeline to assess any modifications
  - provide quality control measures
    - overlay images are produced with the final image mask, defined circles and plate/circle identifiers. 
      - These can be produced from a subset of images to verify the suitability of existing thresholds
  - automate annotation
    - The existing annotation pipeline is designed to handle our specific plate layout but can be readily adapted to other configurations.

Typical image processing time is ~3s per image (48 leaf discs per image) 
and multiple images can be concurrently assessed (multiprocessing).
This also requires less resources than the alternative pre-processing steps (to be quantified) 
and can be performed on the Raspberry Pi directly (to be verified and timed).
Many atypical images are readily detected during circle and plate detection, 
and the availability of data for each image (rather than an aggregate images) 
improves detection of other anomalies and provides opportunity to account for outliers.  

Steps:

    1. Blue rings are identified by Hough Circle transform in median blur of b channel of Lab colourspace (OpenCV)
    2. Clusters of circles are identified to define plates and remove artifact circles (Scipy.cluster.hierarchy)
    3. Orientation and arrangement of plates is determined to define well identities
    4. An image mask is generated from segmentation in both HSV and Lab colour spaces with additional noise removal.
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
    5. Calculate the area of the image mask within each annotated circle and write to file




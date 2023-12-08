# AlGrow

- Alpha-hull colour boundary in segmentation of multiplexed plant and algal images for growth rate analysis.

## The short story 

Algrow is a software tool for automated image annotation, segmentation and analysis. 
It was developed by [Dr Marcus McHale](https://github.com/marcusmchale) 
to support macro-algal disc and plant growth phenotyping in the 
[Plant Systems Biology Laboratory](https://sulpice-lab.com/) of [Dr Ronan Sulpice](https://www.nuigalway.ie/our-research/people/natural-sciences/ronansulpice/) 
at the [University of Galway](https://www.universityofgalway.ie/). 

Features:
  - Improved deterministic model for colour-based image segmentation
    - Hulls (convex or alpha) flexibly define the target colour volume in a given 3D colour space
      - Fixed thresholds are implicitly cubic selections
      - Target means/kMeans are implicitly spherical selections
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

## Get started
apt install python3.10
apt install python3.10-venv

### Install a chosen distribution
Download 
  - get the latest [dist](https://github.com/marcusmchale/algrow/dist)
Install
```pip install algrow.whl```
Run
```./algrow.py```


### Run from source
Download
```git clone https://github.com/marcusmchale/algrow```
Set up virtual environment and activate it (recommended)
```
python3 -m venv venv
. ./venv/bin/activate
```
Install requirements
```
pip install -r REQUIREMENTS.txt
```
Run
```./algrow.py```


### Buidl from Wheel
Download
```git clone https://github.com/marcusmchale/algrow```
Install python
```sudo apt install python3.10 python3.10-venv python3.10-distutils python3.10-dev```
Set up virtual environment (recommended)
```
python3.10 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install dist/algrow-0.3-py3-none-any.whl
```
Build
```python3 -m build```

### PyInstaller
Make sure to include licenses for all dependencies if packaging a binary for distribution.
#### On linux
Create the relevant virtual environment and make sure pyinstaller is installed
```
python3 -m venv venv
. ./venv/bin/activate
pip install -r REQUIREMENTS.txt
pip install pyinstaller
```
Install the following to the system
```
sudo apt install libspatialindex-dev
```
Then run pyinstaller in the algrow root path
You might want to check the path of libspatialindex files
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN \
--name algrow_0_5_0_linux \
--add-data=bmp/logo.png:./bmp/ \
--add-data=venv/lib/python3.10/site-packages/open3d/libc++*.so.1:. \
--add-data=venv/lib/python3.10/site-packages/Rtree.libs/libspatialindex-91fc2909.so.6.1.1:. \
--add-data=venv/lib/python3.10/site-packages/open3d/resources:./open3d/resources \
--add-data=/lib/x86_64-linux-gnu/libspatialindex*:. \
--hidden-import='PIL._tkinter_finder' \
algrow.py
```
#### On macosx
install python 3.10 then create venv with pyinstaller as above
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN \
--name algrow_0_5_0_osx \
--icon=./bmp/icon.ico \
--add-data=bmp/logo.png:./bmp/ \
--add-data=venv/lib/python3.10/site-packages/open3d/resources:./open3d/resources \
--hidden-import='PIL._tkinter_finder' \
algrow.py
``` 
#### On windows
might need to install MS visual c++ redistributable, might be fine if install msvc-runtime before installing open3d
to assess on a fresh system
description:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
the exe: 
https://aka.ms/vs/17/release/vc_redist.x86.exe

In admin powershell(or cmd prompt) run the below to allow script execution,
this is needed at least to activate the venv, 
I didn't test if it is required subsequently as I left it activated.

```set-executionpolicy RemoteSigned```


install python 3.10 (through the app store works)
then in a normal windows cmd, activate the venv and install everything

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install msvc-runtime 
pip install pyinstaller
pip install -r REQUIREMENTS.txt


```
Then run pyinstaller
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN 
--name algrow_0_5_0_win10 
--icon=.\bmp\icon.ico 
--add-data=bmp\logo.png:.\bmp\ 
--add-data=venv\lib\site-packages\open3d\resources:.\open3d\resources
algrow.py
```


#### Icon
to prepare icon file:
```convert -density 384 icon.svg -define icon:auto-resize icon.ico```

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
due to high contrast holding rings (now paint-marker applied to the apparatus)
but also the circular surface of typical plant pots.
We then cluster these circles into plates/trays and assign indices based on relative positions. 
Importantly, this method of relative indexation supports movement of plates and trays across an image series, 
replacing a previously time-consuming process.

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
    - kmeans example  https://www.sciencedirect.com/science/article/pii/S0167865519302806?via%3Dihub 
    - could be doing this but prefer supervised as kmeans  for a single target cluster
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
  - Change area_file/outdir handling to have distinct values for input/output
    - this is important for user experience in the GUI, it is confusing currently
    - could consider loading to args only when run...this would keep them distinct

  - Evaluate/test
    - use kmseg to generate ground truth images and evaluate classification accuracy, naiive kmseg vs corrected with algrow.
    - use coverage similarly
    - test across images with the trained hull. c.f. ML. classifiers.
    - test interaction between true-mask and layout-mask

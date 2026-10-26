# Lane Line Extractor (OpenCV)

A simple lane line extractor built with classical computer vision. It finds white and yellow lane markings in static road images using OpenCV, with no deep learning.

## How it works

The script runs each image through these steps:

| Step | What it does | OpenCV function |
|------|--------------|-----------------|
| 1. Load | Reads the road image | `cv2.imread` |
| 2. HSV filter | Keeps only white and yellow pixels | `cv2.cvtColor`, `cv2.inRange` |
| 3. ROI mask | Keeps only the road area (a trapezoid) and removes sky, trees and mountains | `cv2.fillPoly`, `cv2.bitwise_and` |
| 4. Gaussian blur | Smooths noise before edge detection | `cv2.GaussianBlur` |
| 5. Canny edges | Finds the lane boundaries | `cv2.Canny` |

Detected lane pixels are painted red on the original image. The green shape shows the region of interest.

## Results

Each image shows every stage: original, HSV filtered, blurred, Canny, ROI edges and final result.

### Example 1
![Pipeline](output/road1_pipeline.jpg)

![Result](output/road1_result.jpg)

### Example 2
![Pipeline](output/road2_pipeline.jpg)

![Result](output/road2_result.jpg)

## Setup

1. Install Python 3 and the libraries:
   ```
   pip install opencv-python numpy
   ```
2. Put your road photos (`.jpg`, `.jpeg`, `.png` or `.bmp`) in the `images` folder.

## Run

```
python main.py
```

This processes every image in `images/` and saves two files per image in `output/`:
- `<name>_pipeline.jpg`: all stages side by side
- `<name>_result.jpg`: the final result

To use other folders or a single file:
```
python main.py --input path/to/image.jpg --output my_results
```

## Project structure

```
.
├── main.py          # the full pipeline
├── images/          # input road photos
├── output/          # results are saved here
└── README.md
```

## Tuning

You can change these values at the top of `main.py`:

- `WHITE` and `YELLOW`: the HSV colour ranges. Widen them if lines are missed, narrow them if too much noise appears.
- The ROI polygon in `roi()`: adjust it to match your camera angle.
- Canny thresholds (`50, 150`) in `process()`.

## Limitations

- Fixed colour thresholds struggle with shadows, night scenes, faded paint and strongly colour-graded photos.
- The ROI is hard-coded for a forward-facing, centred view.
- Faint or thin white lines may be missed when the white threshold is strict.
- The output is edge pixels only. Hough transform line fitting would be a natural next step.

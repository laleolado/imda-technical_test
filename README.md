# imda-technical_test
for ai scientist role 2025

# Captcha Solver

This project implements a Python-based Captcha solver using a **pattern matching approach** to identify characters in 5-character image captchas. The core methodology involves:
*   **Template Creation**: Building representative "character maps" (templates) for each unique alphanumeric character (A-Z, 0-9) encountered in a designated training dataset.
*   **Image Preprocessing**: Converting input captcha images into a simplified binary format (black and white) to isolate character pixels.
*   **Character Segmentation**: Isolating individual characters from the preprocessed image.
*   **Matching and Inference**: Comparing each segmented character from an unseen captcha against the stored character templates to find the best match, thereby inferring the captcha text.

The system preprocesses images, builds these character representations from a training set, and then infers characters from new (unseen) captcha images, reporting its accuracy.

## How it Works: Overall Flow

The solver operates in several stages when `captcha_solver.py` is executed:

1.  **Data Collection & Preparation (in `if __name__ == '__main__':`)**:
    *   Scans `sampleCaptchas/input/` for images and `sampleCaptchas/output/` for corresponding true label files.
    *   Filters for valid samples (image found, label is 5 alphanumeric characters).
    *   Splits collected samples into **training set** (e.g., 80%) and **test set** (e.g., 20%).
    *   Attempts to adjust split: ensures training set contains all 36 expected alphanumeric characters (A-Z, 0-9) by moving samples from test set if feasible, while preserving a viable test set.
    *   Checks and warns if training set (after potential enrichment) still lacks any expected characters.

2.  **Training Phase (initiated by `if __name__ == '__main__':`)**:
    *   `Captcha` object initialized.
    *   `train()` method called with **training set** images and labels.
    *   Involves:
        *   Preprocessing each training image (isolates character pixels).
        *   Segmenting individual characters.
        *   Normalizing segmented characters to standard size (dimensions dynamically determined from training data).
        *   Building `self.char_maps` (dictionary of character maps): For each character (A-Z, 0-9) in training set, its representative 2D binary pattern is created using the **first encountered normalized instance**.
    *   **Note on Validation Set**: A separate validation set is not explicitly used during this phase due to:
        *   The small total sample size (e.g., 25 images as initially described for the problem context).
        *   The primary goal being to demonstrate a working solution based on the provided samples, with a clear train/test split for evaluation (the test set serves to measure generalization).
        *   Introducing a validation set would further reduce the already small training set, potentially hindering the creation of comprehensive character maps, especially given the need to represent all 36 alphanumeric characters. The focus is on building the best possible character representations from the available training data.

3.  **Evaluation Phase (in `if __name__ == '__main__':`)**:
    *   Trained `Captcha` solver evaluated using **test set** (unseen during training).
    *   For each test set image:
        *   `__call__` method invoked to infer characters.
        *   Inferred 5-character string compared against true label.
        *   Individual inference results saved to `sampleCaptchas/output_inferred/`.
    *   Finally, **character-level** and **captcha-level accuracy** calculated and printed.

## Detailed Code Explanation (`captcha_solver.py`)

### `Captcha` Class

*   **`__init__(self)`**
    *   Initializes the solver.
    *   `self.char_maps`: Empty dict for character-to-2D NumPy array (character map) mappings. Populated by `train`.
    *   `self.char_height`, `self.char_width`: Dimensions for standardized character maps. Determined dynamically by `train`.
    *   `self.threshold = (50, 50, 50)`: Default RGB threshold for image binarization.
    *   `self.training_samples`: `None` initially. Can be set by `train` if samples passed directly.
    *   `self.sample_captcha_dir`: Default "sampleCaptchas/input". Used by `_train` if `training_samples` not provided.
    *   `self.sample_labels_dir`: Default "sampleCaptchas/output". Used by `_train` if `training_samples` not provided.
    *   **Note**: Sets initial parameters. `train()` must be called explicitly to train.

*   **`train(self, training_samples=None, sample_captcha_dir=None, sample_labels_dir=None)`**
    *   Orchestrates training by calling internal `_train()`.
    *   `training_samples` (optional): List of `(image_path, label_text)` tuples. If provided, these specific samples used for training, overriding directory scanning.
    *   `sample_captcha_dir` (optional): Path to training CAPTCHA image directory. Updates `self.sample_captcha_dir`. Used by `_train` if `training_samples` not provided.
    *   `sample_labels_dir` (optional): Path to training label file directory. Updates `self.sample_labels_dir`. Used by `_train` if `training_samples` not provided.
    *   Actual training logic (segmentation, normalization, map creation) in `_train()`.

*   **`__call__(self, im_path, save_path)`**
    1.  **Check Training**: If model not trained (`self.char_maps` empty), writes error to `save_path`, returns error string.
    2.  **Preprocess**: Calls `_preprocess_image` on `im_path`. If fails, error written, error string returned.
    3.  **Segment**: Calls `_segment_characters` on binarized image.
    4.  **Handle Segmentation Issues**: 
        *   If not exactly 5 characters segmented, constructs `result_text` (pads with "?", takes first 5, or defaults to "?????").
    5.  **Match Characters**: If 5 segments found/adjusted, iterates, calls `_match_character` for each, concatenates results to `result_text`.
    6.  **Save Result**: Writes `result_text` to `save_path`. Ensures output directory exists.
    7.  **Return Value**: Returns inferred `result_text` string.

*   **`_preprocess_image(self, im_path)`**
    1.  **Load Image**: Opens image (Pillow), converts to RGB. Handles file not found/corrupt image by returning `(None, None)` (skipped in training, error in inference via `__call__`).
    2.  **To NumPy Array**: Converts image to NumPy array.
    3.  **Intensity Thresholding**: 
        *   Separates R, G, B channels.
        *   Interpretation: "pixels lighter than (50,50,50) may be re-painted to white - and all other pixels may be marked as black." Implies if pixel *not* purely light (not all R,G,B components > thresholds), it's a character pixel.
        *   Creates boolean `mask`: `True` for character pixels via `(r <= threshold_r) OR (g <= threshold_g) OR (b <= threshold_b)`. Captures character strokes not uniformly black.
    4.  **Binarization**: 
        *   Creates `binary_image_np` (zeros).
        *   Sets pixels to `1` (foreground/character) where `mask` is `True`.
    5.  **Return**: Pillow `Image` (grayscale) and binarized NumPy array (used for next steps).

*   **`_segment_characters(self, binary_image_np, num_chars=5)`**
    1.  **Initial Crop (Content Bounding Box)**:
        *   Calculates column/row sums of `binary_image_np`.
        *   Finds first/last non-empty columns/rows for bounding box of all characters. Removes excess empty space.
        *   Crops `binary_image_np`. If cropped image empty, returns empty list.
    2.  **Valley-Seeking Segmentation**: 
        *   Calculates vertical projection (column sums) of cropped content.
        *   `approx_char_width` = total content width / `num_chars`.
        *   Iterates `num_chars` times:
            *   For first `num_chars-1` characters: 
                *   Defines search window for character end (potential split point), typically around `current_x + approx_char_width`.
                *   Finds column with min pixel sum ("valley") in window = split point.
                *   If window invalid/small, falls back to `approx_char_width` for split.
            *   Last character: takes remaining width.
            *   Segment extracted based on `current_x` and `split_x`.
    3.  **Individual Character Tight Cropping**: 
        *   For each segment from valley-seeking:
            *   If empty/no character pixels, empty array stored.
            *   Else, tight crop by finding its specific non-empty rows/columns. Isolates character precisely.
            *   Appends final tightly cropped 2D NumPy array to `extracted_chars_np`.
    4.  **Return**: `extracted_chars_np` (list of 2D NumPy arrays for segmented characters). May contain empty arrays `[]`.

*   **`_train(self)`**
    1.  **Determine Training Data**:
        *   If `self.training_samples` (from `__init__`) available, uses this list.
        *   Else (legacy, not main script path), scans `self.sample_captcha_dir` and `self.sample_labels_dir`.
    2.  **Iterate and Process Samples**: For each training sample:
        *   **Label File Path**: Constructs path (e.g., `inputXX.jpg` -> `outputXX.txt`). Skips if label missing or not 5 chars.
        *   **Read Label**: Reads 5-char string.
        *   **Preprocess & Segment**: Calls `_preprocess_image`. If fails, skips. Then calls `_segment_characters`.
        *   **Store Segments Temporarily**: If 5 chars segmented, stores each (2D NumPy array) in `temp_char_segments` (dict: char_label -> list of segments).
        *   **Collect Dimensions**: Appends shape (height, width) of valid segments to `all_char_dims`.
    3.  **Determine Standard Character Dimensions**:
        *   If `all_char_dims` not empty:
            *   `self.char_height`: Median of all segment heights.
            *   `self.char_width`: 90th percentile of all segment widths (accommodates varying widths like 'I' vs 'W', trims outliers).
    4.  **Create Character Maps (`self.char_maps`)**: 
        *   Iterates `temp_char_segments`.
        *   For each character, **first raw segment encountered** (`segments_list[0]`) is representative.
        *   Representative raw segment normalized (resized to `self.char_height`, aspect ratio maintained for width, then padded/cropped to `self.char_width`) -> final 2D binary template.
        *   **Store Map**: Stores normalized map in `self.char_maps` (char_label as key).
    5.  **Logging**: Prints status or warnings if no maps created.

*   **`_match_character(self, char_np_segment)`**
    1.  **Handle Empty/Untrained**: If `self.char_maps` empty or `char_np_segment` empty, returns "?".
    2.  **Normalize Input Segment**: Input `char_np_segment` (from inference image) undergoes *same normalization* as training segments (resized to `self.char_height`, padded/cropped to `self.char_width`) -> `processed_segment`.
    3.  **Compare with Stored Maps**: 
        *   Iterates `char_label`, `char_map_template` in `self.char_maps`.
        *   **Calculate Difference**: `np.sum(np.abs(processed_segment - char_map_template))` (Hamming distance for binary arrays).
        *   **Find Best Match**: Tracks `char_label` with `min_diff`.
    4.  **Return**: `best_match_char`.

### `if __name__ == '__main__':` Block

*   Orchestrates: data loading, splitting, training, evaluation, accuracy reporting.
*   **Setup**: Defines paths for `sampleCaptchas/input/` (all images), `sampleCaptchas/output/` (all labels), `sampleCaptchas/output_inferred/` (test inference results).
*   **Data Collection**:
    *   Scans `sample_input_dir` for images (`.jpg`, `.jpeg`, `.png`) and `sample_labels_dir` for `.txt` labels (flexible naming).
    *   Collects valid image-label pairs.
*   **Train/Test Split**:
    *   Shuffles samples (fixed random seed for reproducibility).
    *   Splits data: training (e.g., 80%), test (e.g., 20%).
    *   Attempts to ensure training set has all 36 chars (A-Z, 0-9). If missing, tries to move suitable samples from test to training (constrained to avoid depleting test set).
    *   Prints sample counts per set after adjustment.
    *   Analyzes final training set labels, warns if any of 36 expected chars missing.
*   **Training**:
    *   Initializes `Captcha` solver, passing *only training set samples* to `__init__` (triggers `_train()` with this data).
*   **Evaluation**:
    *   If test set not empty and training successful (char maps created):
        *   Iterates each test sample.
        *   Calls `solver()` (`Captcha.__call__`) for test image. Inferred text returned.
        *   Saves inferred text to `sampleCaptchas/output_inferred/` (e.g., `inputXX_TEST_inferred.txt`).
        *   Compares inferred with true label.
        *   Tracks correct characters and full captcha predictions.
*   **Accuracy Reporting**:
    *   Calculates and prints overall **Captcha-level Accuracy**.
    *   Calculates and prints overall **Character-level Accuracy**.
*   Script now performs systematic evaluation, not just single image inference.

## Solving New (Unseen) Captchas

Primary purpose: Create model to identify characters in new (unseen during training) captcha images.
Once `Captcha` object initialized and `train()` called (populating `solver.char_maps`), `__call__(self, im_path, save_path)` is entry point for solving new CAPTCHAs.

How to use a trained `solver` object for new CAPTCHAs:

```python
# Example: Using a trained Captcha solver for a new, unseen image.

from captcha_solver import Captcha # Assuming class in captcha_solver.py
import os

# --- 1. Prepare Training Data (if not relying on default 'sampleCaptchas') ---
# Option A: List of (image_path, label) tuples (most direct)
# Ensure correct image paths and 5-character string labels.
my_training_samples = [
    ("path/to/your/training_data/train_img1.jpg", "TRN01"),
    ("path/to/your/training_data/train_img2.png", "CAP5A"),
    # ... add more
]

# Option B: Default directory structure (sampleCaptchas/input & sampleCaptchas/output)
# Ensure training images/labels in these folders as per "Directory Structure".
# train() with no args or training_samples=None uses these defaults.


# --- 2. Initialize and Train Captcha Solver ---
print("Initializing and training Captcha solver...")
solver = Captcha()

# Train:
# If Option A (explicit list):
# solver.train(training_samples=my_training_samples)

# If Option B (default dirs):
# Set up dirs first. Solver defaults to "sampleCaptchas/input" & "sampleCaptchas/output".
# For custom training data paths:
# solver.train(sample_captcha_dir="path/to/custom_input_dir", 
#              sample_labels_dir="path/to/custom_output_dir")
# Or, if data in default locations:
solver.train() # Uses self.sample_captcha_dir & self.sample_labels_dir

# After training, check if char_maps created
if not solver.char_maps:
    print("Error: Training failed or no char maps. Cannot solve new CAPTCHAs.")
    # exit() # Or handle error
else:
    print(f"Training complete. {len(solver.char_maps)} char maps ready.")

    # --- 3. Solve a New (Unseen) CAPTCHA ---
    new_captcha_image_path = "path/to/your/new_unseen_captcha.jpg" # Replace
    output_file_for_inferred_text = "path/to/save/inferred_text_for_new_image.txt" # Replace

    # Create output dir if needed
    output_dir = os.path.dirname(output_file_for_inferred_text)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(new_captcha_image_path):
        print(f"Attempting to solve CAPTCHA: {new_captcha_image_path}")
        # __call__ handles preprocessing, segmentation, matching
        inferred_text = solver(new_captcha_image_path, output_file_for_inferred_text)
        
        if inferred_text and not inferred_text.startswith("Error:"):
            print(f"  Successfully inferred text: {inferred_text}")
            print(f"  Prediction saved to: {output_file_for_inferred_text}")
        elif inferred_text: # Error message from solver
            print(f"  Solver returned an error: {inferred_text}")
            print(f"  Error details saved to: {output_file_for_inferred_text}")
        else: # Should not happen
            print(f"  Inference failed for unknown reason for {new_captcha_image_path}.")
    else:
        print(f"Error: New CAPTCHA image not found at {new_captcha_image_path}")

# To run main script (python captcha_solver.py):
# Performs its own train/test split using data from 'sampleCaptchas/'
# and then evaluates. Above example shows programmatic use of Captcha class.
```

The `captcha_solver.py` script demonstrates handling unseen data: trains on a subset (training set from `sampleCaptchas/`), evaluates on a separate "unseen" subset (test set). `__call__` processes any CAPTCHA image path using knowledge from `train()`.

## Setup and Dependencies

Uses Conda for environment management.

1.  **Prerequisites**:
    *   Anaconda or Miniconda installed ([anaconda.com](https://www.anaconda.com/products/distribution)).
    *   Python 3.7+ (env file specifies version).

2.  **Create Conda Environment (`imda_ta`)**:
    *   From `environment.yml`:
      ```bash
      conda env create -f environment.yml
      ```
    *   Or manually:
      ```bash
      conda create --name imda_ta python=3.9  # Or preferred Python 3.x
      conda activate imda_ta
      conda install -c anaconda pillow numpy
      ```

3.  **Activate Environment**: 
    *   Before running script:
      ```bash
      conda activate imda_ta
      ```

*   **Libraries Used**:
    *   Pillow (PIL Fork) - Image manipulation.
    *   NumPy - Numerical operations (array manipulation).

## Directory Structure

Expected structure:

```
imda-technical_test/
|-- captcha_solver.py           # Main solver script
|-- README.md                   # This file
|-- environment.yml             # Conda environment definition
|-- sampleCaptchas/
|   |-- input/                  # ALL sample captcha images (e.g., input00.jpg)
|   |   |-- input00.jpg
|   |   |-- input01.jpg         # .jpg, .jpeg, .png supported
|   |   |-- ...
|   |-- output/                 # TRUE LABELS for ALL sample captchas
|   |   |-- output00.txt  (5-char label for input00.jpg)
|   |   |-- output01.txt
|   |   |-- ...
|   |-- output_inferred/        # INFERRED TEXT for TEST SET captchas saved here
|       |-- inputXX_TEST_inferred.txt # Example result filename
|       |-- ...
```
**Note**: Script processes `.jpg`, `.jpeg`, `.png` images in `input/`.

## Running the Code

1.  **Activate Conda Environment**:
    ```bash
    conda activate imda_ta
    ```
2.  **Run Solver Script**:
    *   Ensure `imda_ta` environment activated.
    *   Run `captcha_solver.py` from command line:
      ```bash
      python captcha_solver.py
      ```
    *   This will:
        1.  Load images/labels from `sampleCaptchas/input/` & `sampleCaptchas/output/`.
        2.  Split data into training and test sets.
        3.  Train `Captcha` model using only training set.
        4.  Evaluate model on test set.
        5.  Print character-level and captcha-level accuracy.
        6.  Save inferred text for each test image to `sampleCaptchas/output_inferred/`.
    *   `if __name__ == '__main__':` block in `captcha_solver.py` controls this.

### Example Output

Output similar to (numbers vary based on `sampleCaptchas` content & random split):

```bash
/path/to/your/project/imda-technical_test> python captcha_solver.py
Loading samples from sampleCaptchas/input and labels from sampleCaptchas/output...
Collected 24 valid samples.
Initial split: 19 training samples, 5 testing samples.
Attempting to enrich training set. Initially missing 1 chars: N
Success: Training set enrichment successfully covered all expected characters.
Final split: 20 training samples and 4 testing samples.
Unique characters in FINAL training set (36 total): 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
Success: The FINAL training set contains all expected 36 characters (A-Z, 0-9).


Initializing Captcha solver...
Training Captcha solver with 20 samples...
Using 20 provided training samples.
Training complete. 36 character maps created: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ

Evaluating on 4 test samples...
Test Sample 1/4: input07.jpg
  True Label:     ZRMQU
  Inferred Label: ZRMQU
Test Sample 2/4: input00.jpg
  True Label:     EGYK4
  Inferred Label: EGYK4
Test Sample 3/4: input03.jpg
  True Label:     J627C
  Inferred Label: J627C
Test Sample 4/4: input20.jpg
  True Label:     Z97ME
  Inferred Label: Z97ME

--- Test Set Evaluation Complete ---
Captcha-level Accuracy: 4/4 = 100.00%
Character-level Accuracy: 20/20 = 100.00%

Script finished.
```

## Assumptions and Limitations

*   **Label Files**: Relies on corresponding `.txt` files in `sample_labels_dir` for initial label collection. Missing/malformed labels (not 5 alphanumeric chars) -> image skipped.
*   **Image File Integrity**: Attempts to skip corrupted/unreadable images.
*   **Label File Naming**: Main script uses flexible naming (e.g., `inputXX.jpg` -> `outputXX.txt` or `image_name.jpg` -> `image_name.txt`). Ensure labels findable.
*   **Training Set Coverage**: Model recognition ability depends on characters present in *training portion* after split. Script warns if chars missing.
*   **Fixed Number of Characters**: Assumes 5 chars per captcha.
*   **Consistent Font/Spacing**: Problem states consistency. Segmentation uses valley-seeking (relies on separable chars), then tight-crops. Significant deviations/overlaps could impact accuracy.
*   **Thresholding**: RGB threshold `(50,50,50)`. Binarization: pixel is character if *any* R,G,B channel <= threshold component (`(r <= 50) OR (g <= 50) OR (b <= 50)`). Aims to preserve faint/colored strokes.
*   **Segmentation Robustness**: Initial content crop, then valley-seeking (vertical pixel projections) for split points. Each segment tight-cropped. More adaptive than fixed-width slicing (which can split chars poorly). Heavy overlap/noise still challenging. (Advanced: connected component analysis - more complex).
    *   Examples of issues with less precise segmentation (resolved by valley-seeking):
        *   `input00.jpg` (EGYK4): 'E' -> 'L' if 'G' part captured.
        *   `input20.jpg` (Z97ME): 'M' -> '1' if segmentation too narrow/poorly centered.
*   **Character Map Quality**: Maps from first encountered normalized segment of each char type in training.
*   **Character Similarity**: Visually similar binarized forms (e.g., 'O' vs '0', 'I' vs '1') can cause misclassifications. Refined thresholding + training data quality/variety are key.

## Rationale for Template Matching Approach

The choice of a template matching approach for this CAPTCHA solver was deliberate, based on the problem's specific characteristics, constraints, and the guidance provided in the technical test description.

**Why Template Matching Was Chosen:**

1.  **Alignment with Problem Statement and Hints**:
    *   The test explicitly stated, "No advanced computer vision background is required" and that "A simple understanding of the 256 x 256 x 256 RGB color space is sufficient." Template matching, which relies on fundamental image processing steps like thresholding, binarization, segmentation, and pixel-wise comparison, directly aligns with this guidance.
    *   The problem description implicitly guided towards such a solution by mentioning the creation of "character maps."

2.  **Suitability for Consistent CAPTCHA Characteristics**:
    *   CAPTCHAs described with highly consistent features: same number of characters, font, spacing, colors/texture, and no skew.
    *   Template matching excels where target objects (characters) have predictable appearance and minimal variations.

3.  **Feasibility with Small Dataset**:
    *   The approach is well-suited for scenarios with a limited number of initial samples (e.g., the 25 sample CAPTCHAs mentioned in the problem context).
    *   Template matching's "training" phase primarily involves creating reference templates from unique characters encountered in the training data, making it robust even with a modest number of examples per character.

4.  **Demonstration of Core Algorithmic Skills**:
    *   Allows demonstration of understanding image manipulation, algorithmic thinking (segmentation, comparison), and structured problem-solving.

**Comparison with Alternative Approaches:**

*   **OCR (Optical Character Recognition) Libraries (e.g., Tesseract)**:
    *   **Pros**: Can be quick to implement and may offer high accuracy for general text recognition.
    *   **Cons for this test**:
        *   Using an off-the-shelf OCR library might not adequately showcase the problem-solving and algorithm development skills the test aimed to assess. It could be seen as sidestepping the core challenge.
        *   Generic OCRs might be overkill or less optimized for the highly specific and consistent nature of these CAPTCHAs compared to a tailored template matching solution.

*   **Deep Learning (DL) Models (e.g., Convolutional Neural Networks - CNNs, Transfer Learning)**:
    *   **Pros**: State-of-the-art for many complex computer vision tasks, can learn intricate features and handle significant variability.
    *   **Cons for this test**:
        *   **Overly Complex for the Task**: Given the problem's simplicity and the direct hints towards a simpler image processing solution, a DL approach could be considered "overthinking."
        *   **Data Hungry**: DL models typically require large datasets for effective training or fine-tuning. The 25 provided samples are insufficient for robust DL model training (even with transfer learning) without significant risk of overfitting.
        *   **Contradicts Simplicity Hint**: DL solution contrasts with "no advanced CV background required" hint.

**Conclusion for This Project:**

The template matching method was selected because it:
*   Directly and effectively addresses the problem as defined.
*   Aligns with provided constraints and guidance.
*   Allows clear demonstration of foundational image processing and algorithmic problem-solving.

While more advanced techniques exist for complex CAPTCHAs, the chosen approach is most appropriate for this technical test.


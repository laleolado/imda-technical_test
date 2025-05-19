# imda-technical_test
for ai scientist role 2025

# Captcha Solver

This project implements a Python-based Captcha solver designed to identify characters in 5-character image captchas. It preprocesses images, builds character representations from a training set, and then infers characters from new (unseen) captcha images, reporting its accuracy.

## How it Works: Overall Flow

The solver operates in several stages when `captcha_solver.py` is executed:

1.  **Data Collection & Preparation (in `if __name__ == '__main__':`)**:
    *   Scans `sampleCaptchas/input/` for images and `sampleCaptchas/output/` for corresponding true label files.
    *   Filters for valid samples (image found, label is 5 alphanumeric characters).
    *   Splits the collected samples into a **training set** (e.g., 80%) and a **test set** (e.g., 20%).
    *   The script then attempts to adjust this split to ensure the training set contains all 36 expected alphanumeric characters (A-Z, 0-9), by moving samples from the test set if necessary and feasible, while trying to preserve a viable test set.
    *   Checks and warns if the training set (after potential enrichment) still does not contain all expected characters (A-Z, 0-9).

2.  **Training Phase (`Captcha` class initialization & `_train` method)**:
    *   A `Captcha` object is initialized using *only* the **training set** images and labels.
    *   This phase involves:
        *   Preprocessing each training image to isolate character pixels.
        *   Segmenting individual characters from these images.
        *   Normalizing the segmented characters to a standard size (dimensions determined dynamically from training data).
        *   Building a dictionary of "character maps" (`self.char_maps`), where each known character encountered in the training set is mapped to its representative 2D binary pattern.

3.  **Evaluation Phase (in `if __name__ == '__main__':`)**:
    *   The trained `Captcha` solver is then evaluated using the **test set** (unseen during training).
    *   For each image in the test set:
        *   The `__call__` method is invoked to infer the characters.
        *   The inferred 5-character string is compared against the true label.
        *   Individual inference results are saved to `sampleCaptchas/output_inferred/`.
    *   Finally, **character-level** and **captcha-level accuracy** are calculated and printed to the console.

## Detailed Code Explanation (`captcha_solver.py`)

### `Captcha` Class

*   **`__init__(self, training_samples=None, sample_captcha_dir="sampleCaptchas/input", sample_labels_dir="sampleCaptchas/output")`**
    *   Initializes the solver.
    *   `training_samples`: An optional list of tuples, where each tuple is `(image_path, label_text)`. If provided, these specific samples are used for training. This is how the `if __name__ == '__main__':` block passes the designated training data to the class.
    *   `sample_captcha_dir`: Path to the directory containing sample captcha images. Used if `training_samples` is `None` (legacy behavior, not primarily used by the main script anymore for training).
    *   `sample_labels_dir`: Path to the directory containing text files with labels for the sample images. Used if `training_samples` is `None`.
    *   `self.char_maps`: An empty dictionary that will store the mapping from a character (e.g., 'A') to its 2D NumPy array representation (the character map).
    *   `self.char_height`, `self.char_width`: Dimensions for the standardized character maps. These are determined dynamically during training.
    *   `self.threshold = (50, 50, 50)`: Default RGB threshold to distinguish character pixels (darker) from background pixels (lighter).
    *   Calls `self._train()` automatically to build the character maps using the determined training data.

*   **`__call__(self, im_path, save_path)`**
    1.  **Check Training**: If the model isn't trained (`self.char_maps` is empty), writes an error to `save_path` and returns `None`.
    2.  **Preprocess**: Calls `_preprocess_image` on the input `im_path`. If preprocessing fails, an error is written to `save_path` and `None` is returned.
    3.  **Segment**: Calls `_segment_characters` on the binarized image.
    4.  **Handle Segmentation Issues**: 
        *   If not exactly 5 characters are segmented, it constructs a `result_text` by either padding with "?" or taking the first 5, or defaulting to "?????".
    5.  **Match Characters**: If 5 segments are found (or adjusted to 5), it iterates through them, calling `_match_character` for each, and concatenates the results to form `result_text`.
    6.  **Save Result**: Writes the `result_text` to the file specified by `save_path`. Ensures the output directory exists.
    7.  **Return Value**: Returns the inferred `result_text` string.

*   **`_preprocess_image(self, im_path)`**
    1.  **Load Image**: Opens the image specified by `im_path` using Pillow (PIL) and converts it to RGB format. Handles potential errors like file not found or unidentifiable/corrupt image by returning `(None, None)` so the image can be skipped during training or cause an error during inference (handled by `__call__`).
    2.  **Convert to NumPy Array**: Transforms the image data into a NumPy array for efficient pixel manipulation.
    3.  **Intensity Thresholding**: 
        *   Separates the R, G, B channels.
        *   Creates a boolean `mask` where `True` indicates pixels that are darker than or equal to `self.threshold` in all three channels (i.e., `(r <= 50) & (g <= 50) & (b <= 50)`).
    4.  **Binarization**: 
        *   Creates a new NumPy array (`binary_image_np`) of the same shape as the mask, initialized to zeros (background).
        *   Sets pixels to `1` (foreground/character) where the `mask` is `True`.
    5.  **Return**: Returns two versions of the processed image: a Pillow `Image` object (grayscale, for potential visualization or other PIL operations) and the binarized NumPy array (which is primarily used for subsequent steps).

*   **`_segment_characters(self, binary_image_np, num_chars=5)`**
    1.  **Initial Crop (Content Bounding Box)**:
        *   Calculates column sums and row sums of the `binary_image_np` (which contains 1s for character pixels and 0s for background).
        *   Finds the first and last non-empty columns and rows to determine the bounding box of all characters together. This removes excess empty space around the entire captcha text.
        *   Crops `binary_image_np` to this bounding box.
    2.  **Character Slot Segmentation**: 
        *   The problem states that font and spacing are the same. This method leverages that by dividing the width of the cropped captcha content by `num_chars` (default 5) to get an average `char_slot_width`.
    3.  **Individual Character Extraction and Cropping**: 
        *   Iterates `num_chars` times:
            *   Slices a vertical segment from the cropped captcha based on the `char_slot_width`.
            *   For this individual character segment, it performs another tight crop by finding its specific non-empty rows and columns. This isolates the character more precisely within its slot.
            *   Appends the final tightly cropped 2D NumPy array of the character to a list `extracted_chars_np`.
    4.  **Return**: Returns `extracted_chars_np`, a list of 2D NumPy arrays, each representing a segmented character.

*   **`_train(self)`**
    1.  **Determine Training Data**:
        *   If `self.training_samples` (passed during `__init__`) is available, it uses this list of image paths and labels directly.
        *   Otherwise (legacy mode, not the primary path for the main script), it scans `self.sample_captcha_dir` for image files and `self.sample_labels_dir` for corresponding labels.
    2.  **Iterate and Process Samples**: For each training sample:
        *   **Determine Label File Path**: Constructs the path to the corresponding label text file. For an image named `inputXX.jpg` in `sample_captcha_dir`, it expects a label file `outputXX.txt` in `sample_labels_dir`. If the label file is missing or the label text is not 5 characters, the image is skipped.
        *   **Read Label**: Reads the 5-character string from the label file.
        *   **Preprocess & Segment**: Calls `_preprocess_image`. If preprocessing fails (e.g., image is corrupt or unreadable), the image is skipped. Then calls `_segment_characters`.
        *   **Store Segments Temporarily**: If 5 characters are successfully segmented, it stores each character segment (2D NumPy array) in `temp_char_segments`, a dictionary where keys are character labels (e.g., 'A') and values are lists of all observed segments for that character.
        *   **Collect Dimensions**: Appends the shape (height, width) of each valid segment to `all_char_dims`.
    3.  **Determine Standard Character Dimensions**:
        *   If `all_char_dims` is not empty (i.e., characters were successfully segmented from samples):
            *   `self.char_height`: Set to the median of all collected segment heights.
            *   `self.char_width`: Set to the 90th percentile of all collected segment widths. This helps accommodate characters of varying widths (like 'I' vs 'W') while providing a standard width for the maps, trimming extreme outliers.
    4.  **Create Character Maps (`self.char_maps`)**: 
        *   Iterates through `temp_char_segments` (each character label and its list of raw segments).
        *   **Select Representative Segment**: For simplicity, it currently takes the *first* segment encountered for each character label as its representative (`best_segment`).
        *   **Normalize Segment**:
            *   Gets the height (`h`) and width (`w`) of `best_segment`.
            *   Resizes it to `self.char_height` while maintaining aspect ratio (calculating `resized_w`). This is done using Pillow's `resize` method with `Image.NEAREST` interpolation suitable for binary images.
            *   Converts the resized PIL image back to a NumPy array of 0s and 1s.
            *   Creates an empty NumPy array (`final_map`) of dimensions (`self.char_height`, `self.char_width`), filled with zeros.
            *   **Padding/Cropping to Standard Width**: The `resized_char_np` is placed into the center of `final_map`. If `resized_w` is less than `self.char_width`, it's padded with zeros. If `resized_w` is greater, it's center-cropped to `self.char_width`. The implementation ensures correct padding/cropping to exactly match the target dimensions.
        *   **Store Map**: Stores this `final_map` in `self.char_maps` with the character label as the key.
    5.  **Logging**: Prints status messages about training completion or warnings if no maps could be created.

*   **`_match_character(self, char_np_segment)`**
    1.  **Handle Empty/Untrained**: If `self.char_maps` is empty or the input `char_np_segment` is empty, returns "?" (unknown).
    2.  **Normalize Input Segment**: The input `char_np_segment` (from an image being inferred) undergoes the *exact same normalization process* as the training segments: resized to `self.char_height` (maintaining aspect ratio for width), then padded/cropped to `self.char_width` to create a `processed_segment`. The implementation ensures correct padding/cropping.
    3.  **Compare with Stored Maps**: 
        *   Iterates through each `char_label` and `char_map_template` in `self.char_maps`.
        *   **Calculate Difference**: Computes the difference between `processed_segment` and `char_map_template`. This is done using `np.sum(np.abs(processed_segment - char_map_template))`, which for binary arrays is equivalent to the Hamming distance (count of differing pixels).
        *   **Find Best Match**: Keeps track of the `char_label` that yields the `min_diff`.
    4.  **Return**: Returns the `best_match_char`.

### `if __name__ == '__main__':` Block

*   This section now orchestrates the entire process of data loading, splitting, training, evaluation, and accuracy reporting.
*   **Setup**: Defines paths for `sampleCaptchas/input/` (all available images), `sampleCaptchas/output/` (all available true labels), and `sampleCaptchas/output_inferred/` (where test set inference results are saved).
*   **Data Collection**:
    *   Scans the `sample_input_dir` for images (`.jpg`, `.jpeg`, `.png`) and `sample_labels_dir` for corresponding `.txt` label files. It uses flexible naming conventions to find labels.
    *   Collects all valid image-label pairs into a list.
*   **Train/Test Split**:
    *   Shuffles the collected samples (with a fixed random seed for reproducibility).
    *   Splits the data into a training set (e.g., 80%) and a test set (e.g., 20%).
    *   The script then attempts to ensure the training set has representatives of all 36 characters (A-Z, 0-9). If characters are missing from the initial training split, it tries to move a few suitable samples (those containing the missing characters) from the initial test set to the training set. This process is constrained to avoid excessively depleting the test set.
    *   Prints the number of samples in each set after this potential adjustment.
    *   Analyzes the final training set labels to identify all unique characters present and warns if any of the 36 expected characters (A-Z, 0-9) are still missing.
*   **Training**:
    *   Initializes the `Captcha` solver, passing *only the training set samples* to its `__init__` method. This triggers the `_train()` method within the class using this specific data.
*   **Evaluation**:
    *   If the test set is not empty and training was successful (character maps were created):
        *   Iterates through each sample in the test set.
        *   Calls the `solver()` (i.e., `Captcha.__call__`) for the test image. The inferred text is returned by `__call__`.
        *   Saves the inferred text to a file in `sampleCaptchas/output_inferred/` (e.g., `inputXX_TEST_inferred.txt`).
        *   Compares the inferred label with the true label.
        *   Tracks correct characters and correct full captcha predictions.
*   **Accuracy Reporting**:
    *   Calculates and prints the overall **Captcha-level Accuracy** (percentage of test captchas correctly identified).
    *   Calculates and prints the overall **Character-level Accuracy** (percentage of individual characters in the test set correctly identified).
*   The script no longer just infers a single image by default but performs a systematic evaluation.

## Solving New (Unseen) Captchas

The primary purpose of this project, as outlined in the technical test, is to create a model that can identify characters in captcha images it has not encountered during training.
Once the `Captcha` object is initialized and trained (which happens automatically when `captcha_solver.py` is run, using the designated training data), its `__call__(self, im_path, save_path)` method is the entry point for this.

You can use a trained `solver` object in your own scripts or for inferring individual new captchas:
```python
# Example of using a trained solver for a new image:
#
# from captcha_solver import Captcha # Assuming your class is in this file
#
# # 1. First, you would need to train a solver instance or load a pre-trained one.
# #    The main script trains it like this:
# #    train_samples = [('path/to/train_image1.jpg', 'ABCDE'), ...] # List of (image_path, label)
# #    solver = Captcha(training_samples=train_samples)
# #
# #    If you run `python captcha_solver.py`, a solver is trained on the split data.
# #    To use *that specific trained instance* externally would require modifying the script
# #    to return or save the solver object.
#
# # 2. For this example, let's assume 'solver' is an already trained Captcha instance:
# #    (If you ran `python captcha_solver.py`, it trains one internally but doesn't expose it directly
# #    without modification. The example below is conceptual if you had a `solver` instance.)
#
# # new_captcha_image_path = "path/to/your/new_unseen_captcha.jpg"
# # output_file_for_inferred_text = "path/to/save/inferred_text.txt"
#
# # if 'solver' in locals() and solver.char_maps: # Check if solver is trained
# #     inferred_text = solver(new_captcha_image_path, output_file_for_inferred_text)
# #     if inferred_text:
# #         print(f"The inferred text for {new_captcha_image_path} is: {inferred_text}")
# #     else:
# #         print(f"Inference failed for {new_captcha_image_path}.")
# # else:
# #     print("Solver is not trained or available in this context.")

```
The main script (`captcha_solver.py`) itself demonstrates the principle of handling unseen data by training on one subset of the available samples (training set) and then evaluating on a separate, "unseen" subset (test set). The `__call__` method is inherently designed to process any captcha image path provided to it, using the character knowledge gained during the `_train` phase.

## Setup and Dependencies

This project uses Conda for environment management.

1.  **Prerequisites**:
    *   Ensure you have Anaconda or Miniconda installed. You can download it from [anaconda.com](https://www.anaconda.com/products/distribution).
    *   Python 3.7+ (the environment file will specify a version).

2.  **Create Conda Environment**:
    *   You can create the Conda environment named `imda_ta` and install the necessary packages using the provided `environment.yml` file:
      ```bash
      conda env create -f environment.yml
      ```
    *   Alternatively, you can create the environment manually and install packages:
      ```bash
      conda create --name imda_ta python=3.9  # Or your preferred Python 3.x version
      conda activate imda_ta
      conda install -c anaconda pillow numpy
      ```

3.  **Activate Environment**: 
    *   Before running the script, always activate the Conda environment:
      ```bash
      conda activate imda_ta
      ```

*   **Libraries Used**:
    *   Pillow (PIL Fork) - For image manipulation.
    *   NumPy - For numerical operations, especially array manipulation.

## Directory Structure

The script expects the following directory structure:

```
imda-technical_test/
|-- captcha_solver.py           # The main solver script
|-- README.md                   # This file
|-- environment.yml             # Conda environment definition
|-- sampleCaptchas/
|   |-- input/                  # Directory for ALL sample captcha images (e.g., input00.jpg)
|   |   |-- input00.jpg
|   |   |-- input01.jpg         # .jpg, .jpeg, .png supported
|   |   |-- ... (other sample images)
|   |-- output/                 # Directory for TRUE LABELS for ALL sample captchas
|   |   |-- output00.txt  (contains the 5-char label for input00.jpg)
|   |   |-- output01.txt  (contains the 5-char label for input01.jpg)
|   |   |-- ...
|   |-- output_inferred/        # Directory where INFERRED TEXT for TEST SET captchas will be saved
|       |-- inputXX_TEST_inferred.txt # Example filename for a test image result
|       |-- ...
```
**Note**: The script processes `.jpg`, `.jpeg`, and `.png` image files found in the `input/` directory.

## Running the Code

1.  **Activate Conda Environment**:
    ```bash
    conda activate imda_ta
    ```
2.  **Run the Solver Script**:
    *   Ensure your Conda environment (`imda_ta`) is activated.
    *   The `captcha_solver.py` script can be run directly from the command line:
      ```bash
      python captcha_solver.py
      ```
    *   This will:
        1.  Load all images and labels from `sampleCaptchas/input/` and `sampleCaptchas/output/`.
        2.  Split this data into a training set and a test set.
        3.  Train a `Captcha` model using only the training set.
        4.  Evaluate the model on the test set.
        5.  Print character-level and captcha-level accuracy to the console.
        6.  Save the inferred text for each test image into the `sampleCaptchas/output_inferred/` directory.
    *   The `if __name__ == '__main__':` block at the end of `captcha_solver.py` controls this behavior.

## Assumptions and Limitations

*   **Label Files**: The solution relies on corresponding `.txt` files in the `sample_labels_dir` for the initial collection of all sample labels. Missing or malformed label files (not 5 alphanumeric characters) will cause the corresponding image to be skipped.
*   **Image File Integrity**: The script will attempt to skip corrupted or unreadable image files.
*   **Label File Naming**: The main script uses flexible naming conventions to find labels (e.g., `inputXX.jpg` -> `outputXX.txt` or `image_name.jpg` -> `image_name.txt`). Ensure your label files can be found.
*   **Training Set Coverage**: The model's ability to recognize specific characters (A-Z, 0-9) is entirely dependent on those characters being present in the *training portion* of the data after the train/test split. If some characters are not included in the training set, the model will not learn them and will likely perform poorly on them if they appear in the test set. The script will warn if characters are missing from the training set.
*   **Fixed Number of Characters**: Assumes all captchas have exactly 5 characters.
*   **Consistent Font/Spacing**: The problem states font and spacing are consistent. The segmentation logic leverages this. Significant deviations could impact segmentation accuracy.
*   **Thresholding**: The default RGB threshold of (50,50,50) might need adjustment if character or background colors vary significantly.
*   **Segmentation Robustness**: While it includes initial content cropping and individual character cropping, the primary segmentation into 5 slots is width-based. More advanced techniques like connected component analysis could be more robust to variations but add complexity.
*   **Character Map Quality**: The character maps are generated from the first encountered segment of each character type during training, after normalization. Averaging or selecting a median representation from multiple samples of the same character could improve map quality but is not currently implemented.
*   **Character Similarity**: If different characters have very similar binarized forms (e.g., 'O' vs '0', 'I' vs '1'), misclassifications can occur. The quality and variety of training samples are key.

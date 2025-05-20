import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import shutil

class Captcha(object):
    def __init__(self):
        """
        Initializes the Captcha solver.
        Sets default parameters. Training data and character maps are initialized as empty
        and are populated by calling the `train` method.
        """
        self.char_maps = {}  # Stores the template for each character
        self.char_height = 0  # Median height of character segments from training
        self.char_width = 0  # Percentile width of character segments from training
        self.threshold = (50, 50, 50)  # RGB threshold for image binarization

        # These will be configured and used by the train method
        self.training_samples = None # List of (image_path, label) tuples for training
        self.sample_captcha_dir = "sampleCaptchas/input"  # Default directory for sample captchas if not using training_samples
        self.sample_labels_dir = "sampleCaptchas/output" # Default directory for sample labels if not using training_samples

    def __call__(self, im_path, save_path):
        """
        Processes a CAPTCHA image and predicts its characters.
        
        Args:
            im_path (str): Path to the CAPTCHA image file.
            save_path (str): Path to save the predicted text (one line).
        
        Returns:
            str: The predicted 5-character string for the CAPTCHA.
                 Returns an error message string (e.g., "Error: ...") if prediction fails at an early stage.
        """
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error: Could not create output directory {output_dir}: {e}. Will attempt to save in current directory.")
                save_path = os.path.basename(save_path)

        # Check if the model is trained
        if not self.char_maps:
            error_msg = "Error: Model not trained (no character maps found)."
            try:
                with open(save_path, 'w') as f:
                    f.write(error_msg + "\n")
            except Exception as e_write:
                 print(f"Error writing error message to {save_path}: {e_write}")
            return error_msg

        # Preprocess the image
        _, binary_image_np = self._preprocess_image(im_path)
        if binary_image_np is None:
            error_msg = f"Error: Failed to preprocess image {im_path}."
            try:
                with open(save_path, 'w') as f:
                    f.write(error_msg + "\n")
            except Exception as e_write:
                print(f"Error writing error message to {save_path}: {e_write}")
            return error_msg

        # Segment characters from the preprocessed image
        char_segments_np = self._segment_characters(binary_image_np)
        
        result_text = ""
        num_segments = len(char_segments_np)

        # Handle cases based on the number of segments found
        if num_segments == 0 and np.sum(binary_image_np) > 0: 
            # If no segments found but image has content, assume 5 unknown chars
            result_text = "?????" 
        elif num_segments != 5:
            # If not exactly 5 segments, attempt to match what was found and pad/truncate with '?'
            if num_segments == 0: # Fully blank or failed segmentation after preprocessing
                 result_text = "?????"
            else: # num_segments is 1, 2, 3, 4, or > 5
                matched_chars_count = 0
                for segment_np in char_segments_np:
                    if matched_chars_count >= 5: # Stop if we already have 5 characters
                        break
                    if segment_np is not None and segment_np.size > 0:
                        result_text += self._match_character(segment_np)
                    else:
                        result_text += "?"
                    matched_chars_count += 1
                
                # Ensure result_text is exactly 5 characters
                if len(result_text) < 5:
                    result_text += "?" * (5 - len(result_text))
                elif len(result_text) > 5: # This case handles num_segments > 5
                    result_text = result_text[:5]
        else: # Exactly 5 segments found
            for segment_np in char_segments_np:
                if segment_np is not None and segment_np.size > 0:
                    result_text += self._match_character(segment_np)
                else:
                    result_text += "?"
        
        # Final length assertion, though prior logic should handle it ensuring 5 chars
        if len(result_text) < 5:
            result_text += "?" * (5 - len(result_text))
        elif len(result_text) > 5:
            result_text = result_text[:5]

        # Save the result
        try:
            with open(save_path, 'w') as f:
                f.write(result_text + "\n")
        except Exception as e:
            print(f"Error writing result to {save_path}: {e}")
        
        return result_text
        
    def train(self, training_samples=None, sample_captcha_dir=None, sample_labels_dir=None):
        """
        Trains the CAPTCHA solver.
        This method populates `self.char_maps` which are used for character recognition.
        It can either use a provided list of (image_path, label) tuples for `training_samples`,
        or scan `sample_captcha_dir` (for images) and `sample_labels_dir` (for .txt labels)
        to gather training data.

        Args:
            training_samples (list, optional): A list of tuples, where each tuple
                is (image_path, label_text). If provided, this is used for training.
                Defaults to None.
            sample_captcha_dir (str, optional): Path to the directory containing training
                CAPTCHA images. Used if `training_samples` is None. If None, uses
                `self.sample_captcha_dir` (which has a default).
            sample_labels_dir (str, optional): Path to the directory containing .txt files
                with labels for the training images. Used if `training_samples` is None.
                If None, uses `self.sample_labels_dir` (which has a default).
        """
        if training_samples is not None:
            self.training_samples = training_samples
        
        # If specific dirs are provided for training, update instance attributes
        # Otherwise, the _train method will use the defaults set in __init__ or previously set ones.
        if sample_captcha_dir is not None:
            self.sample_captcha_dir = sample_captcha_dir
        if sample_labels_dir is not None:
            self.sample_labels_dir = sample_labels_dir
        
        self._train() # Call the internal training logic

    def _preprocess_image(self, im_path):
        """
        Loads an image, converts it to RGB, and then binarizes it based on `self.threshold`.
        Pixels darker than or equal to the threshold are considered foreground (1), others background (0).
        
        Args:
            im_path (str): Path to the image file.
            
        Returns:
            tuple: (PIL.Image.Image, numpy.ndarray)
                - The binarized image as a PIL Image object (black and white).
                - The binarized image as a NumPy array (0s and 1s).
                Returns (None, None) if the image cannot be opened or processed.
        """
        try:
            img = Image.open(im_path).convert('RGB')
        except FileNotFoundError:
            return None, None
        except Exception as e:
            return None, None

        img_np = np.array(img)
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        
        mask = (r <= self.threshold[0]) | (g <= self.threshold[1]) | (b <= self.threshold[2])
        
        binary_image_np = np.zeros(mask.shape, dtype=np.uint8)
        binary_image_np[mask] = 1
        
        binary_pil_img = Image.fromarray(binary_image_np * 255, 'L')

        return binary_pil_img, binary_image_np

    def _segment_characters(self, binary_image_np, num_chars=5):
        """
        Segments individual characters from a binarized CAPTCHA image.
        It first crops the image to the content (bounding box of all black pixels).
        Then, it attempts to find split points between characters. This is done by
        calculating an approximate character width and then searching for a "valley"
        (column with fewest black pixels) within a window around the expected split point.

        Args:
            binary_image_np (numpy.ndarray): The binarized CAPTCHA image (0s for background, 1s for foreground).
            num_chars (int): The expected number of characters in the CAPTCHA (default is 5).

        Returns:
            list: A list of NumPy arrays. Each array is a binarized, tightly cropped segment
                  of an individual character. The list will contain `num_chars` elements.
                  If a character cannot be segmented or is empty, an empty NumPy array is
                  placed at its position. Returns an empty list if the input image is empty
                  or initial cropping fails.
        """
        if binary_image_np is None or binary_image_np.size == 0:
            return []

        col_sums = np.sum(binary_image_np, axis=0)
        row_sums = np.sum(binary_image_np, axis=1)

        if np.sum(col_sums) == 0:
            return []

        first_col_indices = np.where(col_sums > 0)[0]
        if len(first_col_indices) == 0: return []
        first_col = first_col_indices[0]
        
        last_col_indices = np.where(col_sums > 0)[0]
        if len(last_col_indices) == 0: return []
        last_col = last_col_indices[-1]

        first_row_indices = np.where(row_sums > 0)[0]
        if len(first_row_indices) == 0: return []
        first_row = first_row_indices[0]

        last_row_indices = np.where(row_sums > 0)[0]
        if len(last_row_indices) == 0: return []
        last_row = last_row_indices[-1]


        cropped_captcha_np = binary_image_np[first_row:last_row+1, first_col:last_col+1]

        if cropped_captcha_np.shape[0] == 0 or cropped_captcha_np.shape[1] == 0:
            return []

        content_height, content_width = cropped_captcha_np.shape
        col_sums = np.sum(cropped_captcha_np, axis=0)

        extracted_chars_np = []
        current_x = 0
        approx_char_width = content_width / num_chars

        for i in range(num_chars):
            if current_x >= content_width:
                extracted_chars_np.append(np.array([]))
                continue

            if i < num_chars - 1:
                # For characters before the last one, try to find an optimal split point (valley)
                search_start_x = int(current_x + approx_char_width * 0.5)
                search_end_x = int(current_x + approx_char_width * 1.5)
                
                # Clamp search window to be within the image bounds
                search_start_x = max(current_x + 1, search_start_x) 
                search_start_x = min(search_start_x, content_width -1)
                search_end_x = min(search_end_x, content_width -1)
                
                if search_start_x >= search_end_x: # If search window is invalid or too small
                    # Fallback: split based on approximate width
                    split_x = min(content_width, int(current_x + approx_char_width))
                else:
                    # Find the column with the minimum sum of pixels (valley) in the search window
                    relevant_col_sums = col_sums[search_start_x:search_end_x+1]
                    if len(relevant_col_sums) > 0:
                        valley_index_in_window = np.argmin(relevant_col_sums)
                        split_x = search_start_x + valley_index_in_window
                    else: # Should not happen if search_start_x < search_end_x and content_width is positive
                        split_x = min(content_width, int(current_x + approx_char_width))
            else: # For the last character, take all remaining width
                split_x = content_width

            char_segment_np = cropped_captcha_np[:, current_x:split_x]
            current_x = split_x

            if char_segment_np.size == 0 or np.sum(char_segment_np) == 0:
                extracted_chars_np.append(np.array([])) 
                continue
            else:
                char_row_sums = np.sum(char_segment_np, axis=1)
                char_col_sums = np.sum(char_segment_np, axis=0)
                
                if np.all(char_col_sums == 0):
                    extracted_chars_np.append(np.array([])) 
                    continue

                char_first_col_idx = np.where(char_col_sums > 0)[0]
                char_last_col_idx = np.where(char_col_sums > 0)[0]
                char_first_row_idx = np.where(char_row_sums > 0)[0]
                char_last_row_idx = np.where(char_row_sums > 0)[0]

                if not (len(char_first_col_idx)>0 and len(char_last_col_idx)>0 and len(char_first_row_idx)>0 and len(char_last_row_idx)>0):
                    extracted_chars_np.append(np.array([]))
                    continue
                
                char_first_col = char_first_col_idx[0]
                char_last_col = char_last_col_idx[-1]
                char_first_row = char_first_row_idx[0]
                char_last_row = char_last_row_idx[-1]
                
                tight_char_np = char_segment_np[char_first_row:char_last_row+1, char_first_col:char_last_col+1]
                if tight_char_np.size == 0:
                     extracted_chars_np.append(np.array([]))
                else:
                    extracted_chars_np.append(tight_char_np)
        
        return extracted_chars_np

    def _normalize_segment_for_map(self, segment_np):
        """
        Normalizes a segmented character to a standard size (`self.char_height`, `self.char_width`).
        This is crucial for template matching in `_match_character`.
        The process involves:
        1. Resizing the segment to `self.char_height` while maintaining aspect ratio.
        2. Creating a new blank map of `self.char_height` x `self.char_width`.
        3. Centering the resized segment onto this blank map. If the resized segment
           is wider than `self.char_width`, it's centered and cropped. If narrower,
           it's centered with padding.

        Args:
            segment_np (numpy.ndarray): The binarized character segment (0s and 1s).
                                       Can be empty or None.

        Returns:
            numpy.ndarray: The normalized character segment as a binary NumPy array (0s and 1s)
                           of dimensions `self.char_height` x `self.char_width`.
                           Returns a zero array of target dimensions if input is invalid/empty
                           or if `self.char_height`/`self.char_width` are not set (e.g., before training).
        """
        if segment_np is None or segment_np.size == 0:
            height = self.char_height if self.char_height > 0 else 10
            width = self.char_width if self.char_width > 0 else 10
            return np.zeros((height, width), dtype=np.uint8)

        if self.char_height == 0 or self.char_width == 0:
            return np.zeros((10, 10), dtype=np.uint8)


        if segment_np.dtype != np.uint8:
            segment_np = segment_np.astype(np.uint8)
        
        if np.max(segment_np) == 1:
            pil_segment_array = segment_np * 255
        else:
            pil_segment_array = segment_np

        try:
            pil_segment = Image.fromarray(pil_segment_array, 'L')
        except Exception as e:
            return np.zeros((self.char_height, self.char_width), dtype=np.uint8)


        h, w = segment_np.shape

        if h == 0 or w == 0:
             return np.zeros((self.char_height, self.char_width), dtype=np.uint8)
        
        resized_w = int(w * (self.char_height / float(h)))
        if resized_w == 0: resized_w = 1

        try:
            resized_pil = pil_segment.resize((resized_w, self.char_height), Image.NEAREST)
        except Exception as e:
            return np.zeros((self.char_height, self.char_width), dtype=np.uint8)

        resized_char_np_temp = np.array(resized_pil)
        resized_char_np = np.zeros(resized_char_np_temp.shape, dtype=np.uint8)
        resized_char_np[resized_char_np_temp > 127] = 1


        final_map = np.zeros((self.char_height, self.char_width), dtype=np.uint8)
        
        current_w = resized_char_np.shape[1]
        if current_w <= self.char_width:
            offset = (self.char_width - current_w) // 2
            final_map[:, offset:offset+current_w] = resized_char_np
        else:
            offset = (current_w - self.char_width) // 2
            final_map[:, :] = resized_char_np[:, offset:offset+self.char_width]
            
        return final_map

    def _train(self):
        """
        Internal training logic. Builds character templates (`self.char_maps`).
        This method is called by the public `train()` method.
        It processes training images by:
        1. Loading images and their labels (either from `self.training_samples` or by scanning
           `self.sample_captcha_dir` and `self.sample_labels_dir`).
        2. Preprocessing and segmenting characters from each training image.
        3. Collecting all successfully segmented characters and their dimensions.
        4. Calculating a median height (`self.char_height`) and a percentile width (`self.char_width`)
           from all segmented characters. These dimensions are used for normalization.
        5. For each unique character label (A-Z, 0-9), it normalizes one representative segment
           (currently the first one encountered) to create a template map. These maps are
           stored in `self.char_maps`.
        """
        temp_char_segments = {} # Stores lists of segmented numpy arrays for each character label
        all_char_dims = [] # Stores (height, width) of all successfully segmented characters

        training_data_to_process = []
        # Prioritize explicitly provided training_samples
        if self.training_samples:
            print(f"Using {len(self.training_samples)} provided training samples.")
            for im_path, label_text in self.training_samples:
                if not os.path.exists(im_path):
                    print(f"Warning: Training image {im_path} not found. Skipping.")
                    continue
                if not isinstance(label_text, str) or len(label_text) != 5:
                    print(f"Warning: Label for {im_path} is not a 5-character string: '{label_text}'. Skipping.")
                    continue
                training_data_to_process.append({'path': im_path, 'label': label_text})
        else:
            print(f"Scanning for sample captchas in: {self.sample_captcha_dir}")
            image_files = glob.glob(os.path.join(self.sample_captcha_dir, "*.jpg")) + \
                          glob.glob(os.path.join(self.sample_captcha_dir, "*.png"))
            if not image_files:
                print(f"Warning: No training images (.jpg, .png) found in {self.sample_captcha_dir}.")

            for im_path in image_files:
                base_name_parts = os.path.splitext(os.path.basename(im_path))
                base_name = base_name_parts[0]
                
                if base_name.startswith("input"):
                    label_file_name = base_name.replace("input", "output", 1) + ".txt"
                else:
                    label_file_name = base_name + ".txt"
                    
                label_path = os.path.join(self.sample_labels_dir, label_file_name)

                if not os.path.exists(label_path):
                    continue
                try:
                    with open(label_path, 'r') as f:
                        label_text = f.read().strip()
                    if len(label_text) != 5:
                        continue
                    training_data_to_process.append({'path': im_path, 'label': label_text})
                except Exception as e:
                    print(f"Warning: Could not read label file {label_path}: {e}. Skipping image {im_path}.")
                    continue
        
        if not training_data_to_process:
            print("Error: No valid training data (images and corresponding 5-char labels) found or loaded.")
            return

        for item in training_data_to_process:
            im_path = item['path']
            label_text = item['label']

            _, binary_image_np = self._preprocess_image(im_path)
            if binary_image_np is None:
                continue

            char_segments_np = self._segment_characters(binary_image_np)

            if len(char_segments_np) == 5:
                for i, segment_np in enumerate(char_segments_np):
                    if segment_np is not None and segment_np.size > 0:
                        if i < len(label_text):
                            char_label = label_text[i]
                            if char_label not in temp_char_segments:
                                temp_char_segments[char_label] = []
                            temp_char_segments[char_label].append(segment_np)
                            all_char_dims.append(segment_np.shape)

        if not all_char_dims:
            print("Warning: No characters were successfully segmented from any training samples. Character maps cannot be built effectively.")
            self.char_height = 20
            self.char_width = 15
            print(f"Warning: Using fallback char_height={self.char_height}, char_width={self.char_width}")
        else:
            heights = [h for h, w in all_char_dims if h > 0 and w > 0]
            widths = [w for h, w in all_char_dims if h > 0 and w > 0]

            if not heights or not widths:
                print("Warning: No valid character dimensions found after segmentation. Using fallback dimensions.")
                self.char_height = 20 
                self.char_width = 15
            else:
                # Use median height and a high percentile (90th) for width to accommodate wider chars like 'W'
                self.char_height = int(np.median(heights))
                self.char_width = int(np.percentile(widths, 90)) 
        
        if self.char_height <=0: self.char_height = 10 
        if self.char_width <=0: self.char_width = 10  

        for char_label, segments_list in temp_char_segments.items():
            if segments_list:
                representative_raw_segment = segments_list[0]
                final_map = self._normalize_segment_for_map(representative_raw_segment)
                self.char_maps[char_label] = final_map

        if self.char_maps:
            print(f"Training complete. {len(self.char_maps)} character maps created: {''.join(sorted(self.char_maps.keys()))}")
        else:
            print("Warning: Training finished, but no character maps were created. The model will not be effective.")
            
    def _match_character(self, char_np_segment):
        """
        Matches a processed character segment against the stored character maps.
        The input segment is first normalized to the standard dimensions (`self.char_height`, `self.char_width`).
        Then, it's compared to each character template in `self.char_maps` using
        the Sum of Absolute Differences (SAD) pixel-wise. The character corresponding
        to the template with the minimum difference is considered the match.

        Args:
            char_np_segment (numpy.ndarray): The binarized character segment to identify.
                                           Can be empty or None.

        Returns:
            str: The best matching character label (e.g., "A", "7").
                 Returns "?" if:
                 - The model is not trained (no `self.char_maps`).
                 - The input `char_np_segment` is empty or None.
                 - No suitable match is found (e.g., difference too high, though current logic picks best).
                 - `self.char_height` or `self.char_width` are zero (implies training issues).
        """
        if not self.char_maps:
            return "?"
        if char_np_segment is None or char_np_segment.size == 0:
            return "?"
        if self.char_height == 0 or self.char_width == 0:
            pass


        processed_segment = self._normalize_segment_for_map(char_np_segment)

        min_diff = float('inf')
        best_match_char = "?"


        if np.sum(processed_segment) == 0 and not (self.char_height == 0 or self.char_width == 0) :
            pass


        for char_label, char_map_template in self.char_maps.items():
            if processed_segment.shape != char_map_template.shape:
                continue 
            
            diff = np.sum(np.abs(processed_segment - char_map_template))

            if diff < min_diff:
                min_diff = diff
                best_match_char = char_label
        
        return best_match_char


if __name__ == '__main__':
    # Configuration for sample data directories
    sample_input_dir = "sampleCaptchas/input"
    sample_labels_dir = "sampleCaptchas/output" 
    inference_output_dir = "sampleCaptchas/output_inferred" 

    # Create inference output directory if it doesn't exist
    if not os.path.exists(inference_output_dir):
        try:
            os.makedirs(inference_output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create inference output directory {inference_output_dir}: {e}. Exiting.")
            exit()

    all_samples = []
    print(f"Loading samples from {sample_input_dir} and labels from {sample_labels_dir}...")
    
    img_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(sample_input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(sample_input_dir, ext.upper())))

    image_files = sorted(list(set(image_files)))


    if not image_files:
        print(f"Fatal: No images found in {sample_input_dir} with extensions {img_extensions}. Cannot proceed.")
        exit()
        
    for im_path in image_files:
        base_name_full = os.path.basename(im_path)
        base_name = os.path.splitext(base_name_full)[0]
        
        label_file_name = ""
        
        potential_label_names = [
            base_name + ".txt",
            base_name.replace("input", "output", 1) + ".txt" if "input" in base_name else None
        ]
        
        found_label_path = None
        for potential_name in potential_label_names:
            if potential_name is None: continue
            current_label_path = os.path.join(sample_labels_dir, potential_name)
            if os.path.exists(current_label_path):
                found_label_path = current_label_path
                break
        
        if not found_label_path:
            continue
            
        try:
            with open(found_label_path, 'r') as f:
                label_text = f.read().strip().upper() 
            if len(label_text) == 5 and all('A' <= char <= 'Z' or '0' <= char <= '9' for char in label_text):
                all_samples.append({'path': im_path, 'label': label_text})
        except Exception as e:
            print(f"Warning: Could not read or process label file {found_label_path}: {e}. Skipping.")

    if not all_samples:
        print("Fatal: No valid samples (image + 5-char alphanumeric label) collected. Please check your 'sampleCaptchas/input' and 'sampleCaptchas/output' directories and naming conventions. Exiting.")
        exit()
    
    print(f"Collected {len(all_samples)} valid samples.")

    random.seed(42) # For reproducible train/test splits
    random.shuffle(all_samples)
    
    # Splitting data into training and testing sets
    train_samples_data = []
    test_samples_data = []

    if len(all_samples) == 1:
        train_samples_data = all_samples
        test_samples_data = []
        print("Warning: Only 1 sample available. Using it for training, no test set.")
    elif len(all_samples) < 5 : 
        num_train = max(1, len(all_samples) - 1)
        train_samples_data = all_samples[:num_train]
        test_samples_data = all_samples[num_train:]
    else: 
        split_idx = int(0.8 * len(all_samples))
        train_samples_data = all_samples[:split_idx]
        test_samples_data = all_samples[split_idx:]

    print(f"Initial split: {len(train_samples_data)} training samples, {len(test_samples_data)} testing samples.")

    all_possible_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    MIN_TEST_SET_SIZE = max(1, int(0.1 * len(all_samples)))
    
    current_train_list = list(train_samples_data)
    current_test_list = list(test_samples_data)

    initial_train_chars = set()
    for item in current_train_list:
        initial_train_chars.update(list(item['label']))
    
    missing_chars = all_possible_chars - initial_train_chars
    
    if missing_chars:
        print(f"Attempting to enrich training set. Initially missing {len(missing_chars)} chars: {''.join(sorted(list(missing_chars)))}")
        
        moved_samples_count = 0
        max_potential_moves = len(current_test_list) - MIN_TEST_SET_SIZE 

        while missing_chars and len(current_test_list) > MIN_TEST_SET_SIZE and moved_samples_count < max_potential_moves :
            sample_moved_in_this_pass = False
            best_sample_to_move_idx = -1
            
            for i, test_sample in enumerate(current_test_list):
                if any(char in missing_chars for char in test_sample['label']):
                    best_sample_to_move_idx = i
                    break 
            
            if best_sample_to_move_idx != -1:
                sample_to_move = current_test_list.pop(best_sample_to_move_idx)
                current_train_list.append(sample_to_move)
                
                current_train_chars_after_move = set()
                for item in current_train_list:
                    current_train_chars_after_move.update(list(item['label']))
                missing_chars = all_possible_chars - current_train_chars_after_move
                
                sample_moved_in_this_pass = True
                moved_samples_count += 1
            
            if not sample_moved_in_this_pass:
                print("Info: No more suitable samples found in test set to cover remaining missing characters, or test set size limit reached.")
                break
        
        if not missing_chars:
            print("Success: Training set enrichment successfully covered all expected characters.")
        else:
            print(f"Info: Training set enrichment finished. Still missing {len(missing_chars)} chars: {''.join(sorted(list(missing_chars)))}")

    train_samples_data = current_train_list
    test_samples_data = current_test_list

    if not train_samples_data:
        print("Fatal: No samples available for training after split and enrichment. Exiting.")
        exit()
    
    print(f"Final split: {len(train_samples_data)} training samples and {len(test_samples_data)} testing samples.")

    train_for_captcha_class = [(item['path'], item['label']) for item in train_samples_data]

    final_training_chars = set()
    for _, label in train_for_captcha_class:
        final_training_chars.update(list(label))
    print(f"Unique characters in FINAL training set ({len(final_training_chars)} total): {''.join(sorted(list(final_training_chars)))}")
    
    EXPECTED_CHARS = 36 # A-Z (26) + 0-9 (10)
    if len(final_training_chars) < EXPECTED_CHARS:
        still_missing_chars = all_possible_chars - final_training_chars
        print(f"Warning: The FINAL training set is still missing {len(still_missing_chars)} characters: {''.join(sorted(list(still_missing_chars)))}")
        print("The model will only learn from characters present in the training set. Accuracy on unseen missing characters will be affected.")
    else:
        print("Success: The FINAL training set contains all expected 36 characters (A-Z, 0-9).")

    # Initialize the Captcha solver instance
    print("\nInitializing Captcha solver...")
    solver = Captcha()

    # Train the solver using the prepared training samples
    # Pass the sample directories as well, in case training_samples is empty and _train needs to scan them.
    print(f"Training Captcha solver with {len(train_for_captcha_class)} samples...")
    solver.train(
        training_samples=train_for_captcha_class,
        sample_captcha_dir=sample_input_dir, 
        sample_labels_dir=sample_labels_dir
    )

    if not test_samples_data:
        print("\nNo samples available for testing. Skipping evaluation.")
    elif not solver.char_maps: # Check if training was successful by looking for char_maps
        print("\nModel training failed or resulted in no character maps. Skipping evaluation on test set.")
    else:
        print(f"\nEvaluating on {len(test_samples_data)} test samples...")
        total_chars_tested = 0
        correct_chars = 0
        total_captchas_tested = 0
        correct_captchas = 0

        for i, test_item in enumerate(test_samples_data):
            test_im_path = test_item['path']
            true_label = test_item['label'] # Assumed to be 5 chars from data loading
            
            test_output_filename = f"{os.path.splitext(os.path.basename(test_im_path))[0]}_TEST_inferred.txt"
            test_save_path = os.path.join(inference_output_dir, test_output_filename)
            
            # solver.__call__ now returns the prediction string OR an error message string
            inferred_output_from_solver = solver(test_im_path, test_save_path)
            
            # For evaluation, we need a 5-char string.
            # If __call__ returned an error (e.g., "Error: ..."), we treat it as "?????".
            # The actual error message is already logged by __call__ and written to test_save_path.
            eval_inferred_label = "?????" # Default to all unknown for scoring if error
            if isinstance(inferred_output_from_solver, str) and not inferred_output_from_solver.startswith("Error:"):
                eval_inferred_label = inferred_output_from_solver # Use the valid prediction
            elif isinstance(inferred_output_from_solver, str) and inferred_output_from_solver.startswith("Error:"):
                # Error was handled and logged by __call__, file contains error. For eval, it's a failure.
                print(f"  Note: Test sample {os.path.basename(test_im_path)} resulted in solver error: {inferred_output_from_solver}")
            # else: inferred_output_from_solver might be None or unexpected, covered by "?????" default


            print(f"Test Sample {i+1}/{len(test_samples_data)}: {os.path.basename(test_im_path)}")
            print(f"  True Label:     {true_label}")
            # Display the label used for evaluation, and the raw output from solver if different (e.g. error message)
            if eval_inferred_label == inferred_output_from_solver:
                print(f"  Inferred Label: {eval_inferred_label}")
            else:
                print(f"  Inferred Label: {eval_inferred_label} (Solver raw output: '{inferred_output_from_solver}')")


            total_captchas_tested += 1
            if eval_inferred_label == true_label: 
                correct_captchas += 1
            
            # Character-level accuracy: true_label is 5 chars.
            # eval_inferred_label should also be 5 chars (either prediction or "?????").
            for j in range(len(true_label)):
                total_chars_tested += 1
                if j < len(eval_inferred_label) and eval_inferred_label[j] == true_label[j]:
                    correct_chars += 1

        if total_captchas_tested > 0:
            captcha_accuracy = (correct_captchas / total_captchas_tested) * 100
            print(f"\n--- Test Set Evaluation Complete ---")
            print(f"Captcha-level Accuracy: {correct_captchas}/{total_captchas_tested} = {captcha_accuracy:.2f}%")
        
        if total_chars_tested > 0:
            char_accuracy = (correct_chars / total_chars_tested) * 100
            print(f"Character-level Accuracy: {correct_chars}/{total_chars_tested} = {char_accuracy:.2f}%")
        
        if total_captchas_tested == 0 and total_chars_tested == 0 and test_samples_data:
             print("\n--- No test samples were successfully processed for evaluation metrics. ---")
        elif not test_samples_data: #This case already handled by "No samples available for testing"
            pass

    print("\nScript finished.") 
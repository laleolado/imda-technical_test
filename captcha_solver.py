import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import shutil # For managing directories if needed

class Captcha(object):
    def __init__(self, training_samples=None, sample_captcha_dir="sampleCaptchas/input", sample_labels_dir="sampleCaptchas/output"):
        self.char_maps = {}
        self.char_height = 0
        self.char_width = 0
        self.threshold = (50, 50, 50) # Default RGB threshold
        
        # Store training samples if provided directly
        self.training_samples = training_samples 
        
        # Keep these for compatibility if training_samples is None (original behavior)
        self.sample_captcha_dir = sample_captcha_dir
        self.sample_labels_dir = sample_labels_dir

        self._train()

    def _preprocess_image(self, im_path):
        try:
            img = Image.open(im_path).convert('RGB')
        except FileNotFoundError:
            # print(f"Warning: Image file not found: {im_path}")
            return None, None
        except Exception as e:
            # print(f"Warning: Could not read image {im_path}: {e}")
            return None, None

        img_np = np.array(img)
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        
        mask = (r <= self.threshold[0]) & (g <= self.threshold[1]) & (b <= self.threshold[2])
        
        binary_image_np = np.zeros(mask.shape, dtype=np.uint8)
        binary_image_np[mask] = 1
        
        # Create a PIL Image from the binary numpy array for resizing
        binary_pil_img = Image.fromarray(binary_image_np * 255, 'L') # L mode for grayscale

        return binary_pil_img, binary_image_np

    def _segment_characters(self, binary_image_np, num_chars=5):
        if binary_image_np is None or binary_image_np.size == 0:
            return []

        # Initial Crop (Content Bounding Box)
        col_sums = np.sum(binary_image_np, axis=0)
        row_sums = np.sum(binary_image_np, axis=1)

        if np.sum(col_sums) == 0: # Empty image after preprocessing
            return []

        first_col_indices = np.where(col_sums > 0)[0]
        if len(first_col_indices) == 0: return [] # Safeguard
        first_col = first_col_indices[0]
        
        last_col_indices = np.where(col_sums > 0)[0]
        if len(last_col_indices) == 0: return [] # Safeguard
        last_col = last_col_indices[-1]

        first_row_indices = np.where(row_sums > 0)[0]
        if len(first_row_indices) == 0: return [] # Safeguard
        first_row = first_row_indices[0]

        last_row_indices = np.where(row_sums > 0)[0]
        if len(last_row_indices) == 0: return [] # Safeguard
        last_row = last_row_indices[-1]


        cropped_captcha_np = binary_image_np[first_row:last_row+1, first_col:last_col+1]

        if cropped_captcha_np.shape[0] == 0 or cropped_captcha_np.shape[1] == 0: # if width or height is zero after crop
            return []

        # Character Slot Segmentation
        content_width = cropped_captcha_np.shape[1]
        char_slot_width = content_width / num_chars
        
        extracted_chars_np = []
        for i in range(num_chars):
            start_x = int(i * char_slot_width)
            end_x = int((i + 1) * char_slot_width)
            # Ensure end_x does not exceed cropped_captcha_np width
            end_x = min(end_x, cropped_captcha_np.shape[1])
            # Ensure start_x is not greater than or equal to end_x, especially for the last character
            if start_x >= end_x :
                if extracted_chars_np and i == num_chars -1: # If it's the last char and others exist
                    # try to take a small slice from the end if possible
                    start_x = max(0, cropped_captcha_np.shape[1] - int(char_slot_width/2) if char_slot_width > 1 else cropped_captcha_np.shape[1]-1)
                    end_x = cropped_captcha_np.shape[1]
                    if start_x >= end_x : # Still bad, append empty
                         extracted_chars_np.append(np.array([]))
                         continue
                else: # Otherwise problem with slotting
                    extracted_chars_np.append(np.array([]))
                    continue


            char_segment_np = cropped_captcha_np[:, start_x:end_x]

            if char_segment_np.size == 0 or np.sum(char_segment_np) == 0: # Empty segment
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
                    extracted_chars_np.append(np.array([])) # Not enough signal to crop
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
        """ Normalizes a given segment to self.char_height and self.char_width """
        if segment_np is None or segment_np.size == 0:
            # print("Debug: _normalize_segment_for_map received empty or None segment_np.")
            # Return an empty map of standard size if input is invalid or dimensions not set
            height = self.char_height if self.char_height > 0 else 10 # Fallback height
            width = self.char_width if self.char_width > 0 else 10   # Fallback width
            return np.zeros((height, width), dtype=np.uint8)

        if self.char_height == 0 or self.char_width == 0:
            # print("Debug: _normalize_segment_for_map found char_height or char_width is 0.")
            # This case implies training didn't set dimensions properly.
            # Fallback to a default small size, though matching will be poor.
            return np.zeros((10, 10), dtype=np.uint8)


        # Convert segment to PIL Image for resizing
        # Ensure segment_np is of a type PIL can handle (e.g., uint8)
        if segment_np.dtype != np.uint8:
            segment_np = segment_np.astype(np.uint8)
        
        # If segment is binary (0 or 1), scale to 0-255 for PIL
        if np.max(segment_np) == 1:
            pil_segment_array = segment_np * 255
        else:
            pil_segment_array = segment_np # Assume it's already in a displayable range if not 0-1

        try:
            pil_segment = Image.fromarray(pil_segment_array, 'L')
        except Exception as e:
            # print(f"Debug: Error creating PIL Image in _normalize_segment_for_map: {e}. Segment shape: {segment_np.shape}, dtype: {segment_np.dtype}, max_val: {np.max(segment_np)}")
            return np.zeros((self.char_height, self.char_width), dtype=np.uint8) # Fallback


        h, w = segment_np.shape # Original dimensions of the numpy array

        if h == 0 or w == 0: # Safeguard if an empty array somehow got this far
             # print("Debug: _normalize_segment_for_map received segment_np with zero dimension.")
             return np.zeros((self.char_height, self.char_width), dtype=np.uint8)
        
        # Resize to self.char_height while maintaining aspect ratio
        resized_w = int(w * (self.char_height / float(h)))
        if resized_w == 0: resized_w = 1 # Ensure width is at least 1

        try:
            # NEAREST is good for binary images, ANTIALIAS or BICUBIC might be better if not strictly binary
            resized_pil = pil_segment.resize((resized_w, self.char_height), Image.NEAREST)
        except Exception as e:
            # print(f"Warning: PIL resize failed in _normalize_segment_for_map. Error: {e}")
            return np.zeros((self.char_height, self.char_width), dtype=np.uint8)

        # Convert back to NumPy array and binarize (ensure values are 0 or 1)
        # The threshold here should ideally be adaptive or based on image properties
        # For now, assuming >127 is foreground after resize from 0-255 range.
        resized_char_np_temp = np.array(resized_pil)
        resized_char_np = np.zeros(resized_char_np_temp.shape, dtype=np.uint8)
        resized_char_np[resized_char_np_temp > 127] = 1


        # Create final map and place the resized character (pad or crop width)
        final_map = np.zeros((self.char_height, self.char_width), dtype=np.uint8)
        
        current_w = resized_char_np.shape[1]
        if current_w <= self.char_width: # Pad
            offset = (self.char_width - current_w) // 2
            final_map[:, offset:offset+current_w] = resized_char_np
        else: # current_w > self.char_width (crop)
            offset = (current_w - self.char_width) // 2
            final_map[:, :] = resized_char_np[:, offset:offset+self.char_width]
            
        return final_map

    def _train(self):
        temp_char_segments = {} # char_label -> list of raw 2D numpy segments
        all_char_dims = [] # list of (h, w) tuples

        training_data_to_process = []
        if self.training_samples: # Use provided list of (image_path, label_text)
            print(f"Using {len(self.training_samples)} provided training samples.")
            for im_path, label_text in self.training_samples:
                if not os.path.exists(im_path):
                    print(f"Warning: Training image {im_path} not found. Skipping.")
                    continue
                if not isinstance(label_text, str) or len(label_text) != 5:
                    print(f"Warning: Label for {im_path} is not a 5-character string: '{label_text}'. Skipping.")
                    continue
                training_data_to_process.append({'path': im_path, 'label': label_text})
        else: # Original behavior: scan directories
            print(f"Scanning for sample captchas in: {self.sample_captcha_dir}")
            image_files = glob.glob(os.path.join(self.sample_captcha_dir, "*.jpg")) + \
                          glob.glob(os.path.join(self.sample_captcha_dir, "*.png"))
            if not image_files:
                print(f"Warning: No training images (.jpg, .png) found in {self.sample_captcha_dir}.")
                # No return here, will proceed and later check if training_data_to_process is empty

            for im_path in image_files:
                base_name_parts = os.path.splitext(os.path.basename(im_path))
                base_name = base_name_parts[0]
                
                # Try to match 'inputXX' -> 'outputXX.txt' convention
                if base_name.startswith("input"):
                    label_file_name = base_name.replace("input", "output", 1) + ".txt"
                else: # If not starting with "input", assume direct name match + .txt
                    label_file_name = base_name + ".txt"
                    
                label_path = os.path.join(self.sample_labels_dir, label_file_name)

                if not os.path.exists(label_path):
                    # print(f"Warning: Label file not found for {im_path} (expected at {label_path}). Skipping.")
                    continue
                try:
                    with open(label_path, 'r') as f:
                        label_text = f.read().strip()
                    if len(label_text) != 5:
                        # print(f"Warning: Label in {label_path} for {im_path} is not 5 characters: '{label_text}'. Skipping.")
                        continue
                    training_data_to_process.append({'path': im_path, 'label': label_text})
                except Exception as e:
                    print(f"Warning: Could not read label file {label_path}: {e}. Skipping image {im_path}.")
                    continue
        
        if not training_data_to_process:
            print("Error: No valid training data (images and corresponding 5-char labels) found or loaded.")
            return # Critical, cannot train

        # print(f"Starting training process with {len(training_data_to_process)} samples...")

        for item in training_data_to_process:
            im_path = item['path']
            label_text = item['label']

            _, binary_image_np = self._preprocess_image(im_path)
            if binary_image_np is None:
                # print(f"Skipping {im_path} due to preprocessing failure.")
                continue

            char_segments_np = self._segment_characters(binary_image_np)

            if len(char_segments_np) == 5:
                for i, segment_np in enumerate(char_segments_np):
                    if segment_np is not None and segment_np.size > 0: # Ensure segment is not empty
                        if i < len(label_text): # Ensure label_text is long enough
                            char_label = label_text[i]
                            if char_label not in temp_char_segments:
                                temp_char_segments[char_label] = []
                            temp_char_segments[char_label].append(segment_np)
                            all_char_dims.append(segment_np.shape) # (height, width)
                        # else:
                            # print(f"Warning: Label text '{label_text}' too short for segment index {i} from image {im_path}")
            # else:
                # print(f"Warning: Did not segment 5 chars from {im_path} (got {len(char_segments_np)}). Label: {label_text}.")


        if not all_char_dims:
            print("Warning: No characters were successfully segmented from any training samples. Character maps cannot be built effectively.")
            # Don't return yet, still need to set default char_height/width if possible,
            # or rely on _normalize_segment_for_map fallbacks.
            self.char_height = 20 # Arbitrary fallback
            self.char_width = 15  # Arbitrary fallback
            print(f"Warning: Using fallback char_height={self.char_height}, char_width={self.char_width}")
        else:
            heights = [h for h, w in all_char_dims if h > 0 and w > 0]
            widths = [w for h, w in all_char_dims if h > 0 and w > 0]

            if not heights or not widths:
                print("Warning: No valid character dimensions found after segmentation. Using fallback dimensions.")
                self.char_height = 20 
                self.char_width = 15
            else:
                self.char_height = int(np.median(heights))
                self.char_width = int(np.percentile(widths, 90)) 
        
        # Ensure minimal dimensions
        if self.char_height <=0: self.char_height = 10 
        if self.char_width <=0: self.char_width = 10  
        # print(f"Determined standard character map size: Height={self.char_height}, Width={self.char_width}")

        # Create Character Maps
        for char_label, segments_list in temp_char_segments.items():
            if segments_list:
                # For simplicity, take the first segment. Could be averaged or median image.
                # To be more robust, one might average or take a median *normalized* segment
                
                # Let's try to find the "best" segment (e.g. largest area, or median size)
                # For now, still using the first one encountered for simplicity of the original logic.
                best_segment_raw = segments_list[0] 
                
                final_map = self._normalize_segment_for_map(best_segment_raw)
                self.char_maps[char_label] = final_map
        
        if self.char_maps:
            print(f"Training complete. {len(self.char_maps)} character maps created: {''.join(sorted(self.char_maps.keys()))}")
        else:
            print("Warning: Training finished, but no character maps were created. The model will not be effective.")
            
    def _match_character(self, char_np_segment):
        if not self.char_maps:
            return "?"
        if char_np_segment is None or char_np_segment.size == 0:
            return "?"
        if self.char_height == 0 or self.char_width == 0:
            # This implies training didn't set dimensions, _normalize will use fallbacks
            # print("Warning: Standard character dimensions not set during _match_character. Normalization may use fallbacks.")
            pass


        processed_segment = self._normalize_segment_for_map(char_np_segment)
        # print(f"Debug: _match_character: processed_segment shape {processed_segment.shape}, sum {np.sum(processed_segment)}")


        min_diff = float('inf')
        best_match_char = "?"

        if np.sum(processed_segment) == 0 and not (self.char_height == 0 or self.char_width == 0) : # if normalized segment is blank
            # print("Debug: Normalized segment is blank, likely won't match anything well.")
            # it will naturally have high diff with non-blank templates
            pass


        for char_label, char_map_template in self.char_maps.items():
            # print(f"Debug: Comparing with char_map_template for '{char_label}', shape {char_map_template.shape}, sum {np.sum(char_map_template)}")
            if processed_segment.shape != char_map_template.shape:
                # This can happen if char_height/width was 0 during map creation for some maps
                # print(f"Warning: Shape mismatch! Processed: {processed_segment.shape}, Map '{char_label}': {char_map_template.shape}. Skipping comparison.")
                continue 
            
            # Hamming distance for binary arrays
            diff = np.sum(np.abs(processed_segment - char_map_template))
            # print(f"  Diff with '{char_label}': {diff}")

            if diff < min_diff:
                min_diff = diff
                best_match_char = char_label
            # Optional: Handle ties or low-confidence matches
            # For example, if min_diff is still very high, could return "?"
        
        # print(f"Debug: Best match for segment: '{best_match_char}' with diff {min_diff}")
        return best_match_char

    def __call__(self, im_path, save_path):
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error: Could not create output directory {output_dir}: {e}. Will attempt to save in current directory.")
                save_path = os.path.basename(save_path)


        if not self.char_maps:
            error_msg = "Error: Model not trained (no character maps found)."
            try:
                with open(save_path, 'w') as f:
                    f.write(error_msg + "\n")
            except Exception as e_write:
                 print(f"Error writing error message to {save_path}: {e_write}")
            return # No result to return, but file indicates error

        _, binary_image_np = self._preprocess_image(im_path)
        if binary_image_np is None:
            error_msg = f"Error: Failed to preprocess image {im_path}."
            try:
                with open(save_path, 'w') as f:
                    f.write(error_msg + "\n")
            except Exception as e_write:
                print(f"Error writing error message to {save_path}: {e_write}")
            return

        char_segments_np = self._segment_characters(binary_image_np)
        
        result_text = ""
        num_segments = len(char_segments_np)

        if num_segments == 0 and np.sum(binary_image_np) > 0: 
            result_text = "?????"
        elif num_segments != 5:
            if num_segments == 0: 
                 result_text = "?????"
            elif num_segments < 5:
                for segment_np in char_segments_np:
                    if segment_np is not None and segment_np.size > 0:
                        result_text += self._match_character(segment_np)
                    else:
                        result_text += "?" # Placeholder for empty/failed segment
                result_text += "?" * (5 - len(result_text)) # Pad to 5 chars
            else: # num_segments > 5
                count = 0
                for segment_np in char_segments_np:
                    if count >= 5: break
                    if segment_np is not None and segment_np.size > 0:
                        result_text += self._match_character(segment_np)
                        count +=1
                    else: # if a segment is bad, still count it as a slot attempt
                        result_text += "?"
                        count +=1
                result_text = result_text[:5] # Ensure it's 5 chars
                if len(result_text) < 5: # if too many bad segments
                    result_text += "?" * (5 - len(result_text))


        else: # Exactly 5 segments
            for segment_np in char_segments_np:
                if segment_np is not None and segment_np.size > 0:
                    result_text += self._match_character(segment_np)
                else:
                    result_text += "?" # Placeholder for empty/failed segment
        
        # Final check to ensure result_text is exactly 5 chars
        if len(result_text) < 5:
            result_text += "?" * (5 - len(result_text))
        elif len(result_text) > 5:
            result_text = result_text[:5]

        try:
            with open(save_path, 'w') as f:
                f.write(result_text + "\n")
        except Exception as e:
            print(f"Error writing result to {save_path}: {e}")
        
        return result_text # Return the text for direct use in evaluation loop


if __name__ == '__main__':
    sample_input_dir = "sampleCaptchas/input"
    sample_labels_dir = "sampleCaptchas/output" 
    inference_output_dir = "sampleCaptchas/output_inferred" 

    # Ensure the directory for saving inferred results exists
    if not os.path.exists(inference_output_dir):
        try:
            os.makedirs(inference_output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error: Could not create inference output directory {inference_output_dir}: {e}. Exiting.")
            exit()

    # 1. Collect all available samples (image path, label string)
    all_samples = []
    print(f"Loading samples from {sample_input_dir} and labels from {sample_labels_dir}...")
    
    # Look for .jpg, .jpeg, .png files (case-insensitive for extensions)
    img_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(sample_input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(sample_input_dir, ext.upper()))) # for .PNG, .JPG

    # Remove duplicates if any (e.g. if both *.png and *.PNG found same file on case-insensitive fs)
    image_files = sorted(list(set(image_files)))


    if not image_files:
        print(f"Fatal: No images found in {sample_input_dir} with extensions {img_extensions}. Cannot proceed.")
        # --- DUMMY DATA CREATION (Comment out if not needed) ---
        # print("Attempting to create dummy data for testing...")
        # dummy_dir = "sampleCaptchas_dummy"
        # dummy_input_dir = os.path.join(dummy_dir, "input")
        # dummy_output_dir = os.path.join(dummy_dir, "output")
        # os.makedirs(dummy_input_dir, exist_ok=True)
        # os.makedirs(dummy_output_dir, exist_ok=True)
        # for i in range(5):
        #     try:
        #         img = Image.new('RGB', (150, 50), color = 'white')
        #         d = ImageDraw.Draw(img)
        #         font = ImageFont.truetype("arial.ttf", 30) # Needs a .ttf font file
        #         text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=5))
        #         d.text((10,10), text, fill=(5,5,5), font=font)
        #         img_path = os.path.join(dummy_input_dir, f"input{i:02d}.png")
        #         img.save(img_path)
        #         with open(os.path.join(dummy_output_dir, f"output{i:02d}.txt"), "w") as f:
        #             f.write(text)
        #         print(f"Created dummy: {img_path} with label {text}")
        #     except Exception as e_dummy:
        #         print(f"Could not create dummy image/label {i}: {e_dummy}. A .ttf font file (e.g. arial.ttf) must be available.")
        #         print("Please ensure you have actual sample captchas or fix dummy data creation.")
        #         exit()
        # print(f"Using DUMMY data from {dummy_dir}. Please replace with actual data for real testing.")
        # sample_input_dir = dummy_input_dir
        # sample_labels_dir = dummy_output_dir
        # image_files = glob.glob(os.path.join(sample_input_dir, "*.png"))
        # if not image_files: exit()
        # --- END DUMMY DATA ---
        exit()
        
    for im_path in image_files:
        base_name_full = os.path.basename(im_path)
        base_name = os.path.splitext(base_name_full)[0]
        
        label_file_name = ""
        # Flexible label name finding:
        # 1. Exact match base_name.txt in sample_labels_dir
        # 2. inputXX -> outputXX.txt
        # 3. If base_name contains "input" try replacing first instance with "output"
        
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
            # print(f"Warning: Label file not found for {im_path} (tried conventions). Skipping.")
            continue
            
        try:
            with open(found_label_path, 'r') as f:
                label_text = f.read().strip().upper() 
            if len(label_text) == 5 and all('A' <= char <= 'Z' or '0' <= char <= '9' for char in label_text):
                all_samples.append({'path': im_path, 'label': label_text})
            # else:
                # print(f"Warning: Invalid label in {found_label_path} for {im_path}: '{label_text}'. Skipping.")
        except Exception as e:
            print(f"Warning: Could not read or process label file {found_label_path}: {e}. Skipping.")

    if not all_samples:
        print("Fatal: No valid samples (image + 5-char alphanumeric label) collected. Please check your 'sampleCaptchas/input' and 'sampleCaptchas/output' directories and naming conventions. Exiting.")
        exit()
    
    print(f"Collected {len(all_samples)} valid samples.")

    # 2. Split into training and testing sets
    random.seed(42) # for reproducible splits
    random.shuffle(all_samples)
    
    train_samples_data = []
    test_samples_data = []

    if len(all_samples) == 1:
        train_samples_data = all_samples
        test_samples_data = []
        print("Warning: Only 1 sample available. Using it for training, no test set.")
    elif len(all_samples) < 5 : 
        num_train = max(1, len(all_samples) - 1) # Ensure at least 1 for training
        train_samples_data = all_samples[:num_train]
        test_samples_data = all_samples[num_train:]
    else: 
        split_idx = int(0.8 * len(all_samples))
        train_samples_data = all_samples[:split_idx]
        test_samples_data = all_samples[split_idx:]

    print(f"Initial split: {len(train_samples_data)} training samples, {len(test_samples_data)} testing samples.")

    # --- Logic to enrich training set with all characters ---
    all_possible_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    MIN_TEST_SET_SIZE = max(1, int(0.1 * len(all_samples))) # Keep at least 10% or 1 sample for testing
    
    # Ensure train_samples_data and test_samples_data are lists of dictionaries
    # (they should be already, but this makes the types explicit for manipulation)
    current_train_list = list(train_samples_data)
    current_test_list = list(test_samples_data)

    # Calculate initial missing characters from the training set
    initial_train_chars = set()
    for item in current_train_list:
        initial_train_chars.update(list(item['label']))
    
    missing_chars = all_possible_chars - initial_train_chars
    
    if missing_chars:
        print(f"Attempting to enrich training set. Initially missing {len(missing_chars)} chars: {''.join(sorted(list(missing_chars)))}")
        
        moved_samples_count = 0
        # Limit moves to prevent excessive test set depletion, e.g., half of the initial test set size
        # or until test set hits MIN_TEST_SET_SIZE.
        max_potential_moves = len(current_test_list) - MIN_TEST_SET_SIZE 

        # Iterate while characters are missing and test set is large enough to take from
        while missing_chars and len(current_test_list) > MIN_TEST_SET_SIZE and moved_samples_count < max_potential_moves :
            sample_moved_in_this_pass = False
            best_sample_to_move_idx = -1
            # Find a sample in test_list that provides one or more *currently* missing characters
            # Prioritize samples that cover more missing characters if complex, but for now, first good one.
            
            # Iterate over a copy for safe removal if iterating by index, or find index then pop.
            for i, test_sample in enumerate(current_test_list):
                if any(char in missing_chars for char in test_sample['label']):
                    best_sample_to_move_idx = i
                    break 
            
            if best_sample_to_move_idx != -1:
                sample_to_move = current_test_list.pop(best_sample_to_move_idx)
                current_train_list.append(sample_to_move)
                
                # Update missing_chars based on the new state of current_train_list
                # This is less efficient than updating initial_train_chars and re-calculating,
                # but clearer for now. Could optimize by adding to initial_train_chars.
                current_train_chars_after_move = set()
                for item in current_train_list:
                    current_train_chars_after_move.update(list(item['label']))
                missing_chars = all_possible_chars - current_train_chars_after_move
                
                sample_moved_in_this_pass = True
                moved_samples_count += 1
                # print(f"Moved sample. Remaining missing: {len(missing_chars)} chars. Test set size: {len(current_test_list)}")
            
            if not sample_moved_in_this_pass:
                # No suitable sample found in test_list to cover remaining missing_chars, or test set too small.
                print("Info: No more suitable samples found in test set to cover remaining missing characters, or test set size limit reached.")
                break
        
        if not missing_chars:
            print("Success: Training set enrichment successfully covered all expected characters.")
        else:
            print(f"Info: Training set enrichment finished. Still missing {len(missing_chars)} chars: {''.join(sorted(list(missing_chars)))}")

    # Update the main variables after enrichment process
    train_samples_data = current_train_list
    test_samples_data = current_test_list
    # --- End of enrichment logic ---

    if not train_samples_data:
        print("Fatal: No samples available for training after split and enrichment. Exiting.")
        exit()
    
    print(f"Final split: {len(train_samples_data)} training samples and {len(test_samples_data)} testing samples.")

    train_for_captcha_class = [(item['path'], item['label']) for item in train_samples_data]

    # This check for missing characters will now run on the potentially enriched training set
    final_training_chars = set()
    for _, label in train_for_captcha_class: # Use the final train_for_captcha_class
        final_training_chars.update(list(label))
    print(f"Unique characters in FINAL training set ({len(final_training_chars)} total): {''.join(sorted(list(final_training_chars)))}")
    
    EXPECTED_CHARS = 36 # A-Z (26) + 0-9 (10)
    if len(final_training_chars) < EXPECTED_CHARS:
        # all_possible_chars is already defined
        still_missing_chars = all_possible_chars - final_training_chars
        print(f"Warning: The FINAL training set is still missing {len(still_missing_chars)} characters: {''.join(sorted(list(still_missing_chars)))}")
        print("The model will only learn from characters present in the training set. Accuracy on unseen missing characters will be affected.")
    else:
        print("Success: The FINAL training set contains all expected 36 characters (A-Z, 0-9).")

    # 3. Initialize and train the Captcha solver using ONLY training data
    print("\nInitializing Captcha solver with training data...")
    solver = Captcha(training_samples=train_for_captcha_class)

    # 4. Evaluate on the test set
    if not test_samples_data:
        print("\nNo samples available for testing. Skipping evaluation.")
    elif not solver.char_maps: # Check if training actually produced maps
        print("\nModel training failed or resulted in no character maps. Skipping evaluation on test set.")
    else:
        print(f"\nEvaluating on {len(test_samples_data)} test samples...")
        total_chars_tested = 0
        correct_chars = 0
        total_captchas_tested = 0
        correct_captchas = 0

        for i, test_item in enumerate(test_samples_data):
            test_im_path = test_item['path']
            true_label = test_item['label']
            
            test_output_filename = f"{os.path.splitext(os.path.basename(test_im_path))[0]}_TEST_inferred.txt"
            test_save_path = os.path.join(inference_output_dir, test_output_filename)
            
            # Perform inference, __call__ now returns the inferred_label
            inferred_label = solver(test_im_path, test_save_path)
            
            # Fallback if __call__ didn't return text (e.g. due to internal error before text generation)
            if inferred_label is None: 
                # Try to read from file as a last resort, though it might contain an error message
                if os.path.exists(test_save_path):
                    try:
                        with open(test_save_path, 'r') as f_read:
                            content = f_read.read().strip()
                        if content.startswith("Error:"): inferred_label = "?????" # Mark as full error if file contains error
                        else: inferred_label = content
                    except Exception:
                        inferred_label = "?????" # Failed to read
                else:
                     inferred_label = "?????" # File not even created

            print(f"Test Sample {i+1}/{len(test_samples_data)}: {os.path.basename(test_im_path)}")
            print(f"  True Label:     {true_label}")
            print(f"  Inferred Label: {inferred_label}")

            total_captchas_tested += 1
            # Ensure inferred_label is a string for comparison
            if isinstance(inferred_label, str) and inferred_label == true_label:
                correct_captchas += 1
                # print("  Captcha: CORRECT")
            # else:
                # print("  Captcha: INCORRECT")
            
            if isinstance(inferred_label, str) and len(inferred_label) == len(true_label):
                for j in range(len(true_label)):
                    total_chars_tested += 1
                    if inferred_label[j] == true_label[j]:
                        correct_chars += 1
            elif isinstance(inferred_label, str): # Length mismatch, count all as incorrect for char accuracy
                total_chars_tested += len(true_label)
                # print(f"  Note: Length mismatch between true ({len(true_label)}) and inferred ({len(inferred_label)}). All chars counted as incorrect for this sample.")
            else: # Inferred label is not a string (e.g. None)
                total_chars_tested += len(true_label) # Count all as incorrect
                # print(f"  Note: Inferred label was not a valid string. All chars counted as incorrect.")


        # 5. Report Accuracy
        if total_captchas_tested > 0:
            captcha_accuracy = (correct_captchas / total_captchas_tested) * 100
            print(f"\n--- Test Set Evaluation Complete ---")
            print(f"Captcha-level Accuracy: {correct_captchas}/{total_captchas_tested} = {captcha_accuracy:.2f}%")
        
        if total_chars_tested > 0:
            char_accuracy = (correct_chars / total_chars_tested) * 100
            print(f"Character-level Accuracy: {correct_chars}/{total_chars_tested} = {char_accuracy:.2f}%")
        
        if total_captchas_tested == 0 and total_chars_tested == 0 and test_samples_data:
             print("\n--- No test samples were successfully processed for evaluation metrics. ---")
        elif not test_samples_data:
            pass # Already handled by "No samples available for testing"

    print("\nScript finished.") 
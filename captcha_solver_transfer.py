print("Script started")

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance # Pillow for image manipulation
import os
import random
from sklearn.model_selection import train_test_split # For splitting data

# Attempt to import TensorFlow and its components
try:
    print("Attempting to import TensorFlow and Keras...")
    import tensorflow as tf
    print(f"Successfully imported TensorFlow version: {tf.__version__}")
    
    # Import Keras directly instead of from tensorflow
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("Successfully imported Keras components")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"Import error: {str(e)}")
    print("Required packages are not installed. The Captcha class will not be fully functional for training and prediction.")
    # Define dummy classes/functions if TensorFlow is not available to allow the script to be parsed
    Model = object
    Dense = GlobalAveragePooling2D = Dropout = BatchNormalization = Input = object
    MobileNetV2 = object
    mobilenet_preprocess_input = lambda x: x # Dummy function
    to_categorical = lambda y, num_classes: y # Dummy function

# --- Constants ---
# Dimensions for individual character images fed into MobileNetV2
CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT = 96, 96 # MobileNetV2 often expects this or larger, and 3 channels
NUM_CHARACTERS_PER_CAPTCHA = 5 # As specified: strictly 5-character captchas
# Allowed characters: 0-9 (10), A-Z (26) = 36 classes
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_CLASSES = len(CHARACTERS)

# Mappings for labels to integers and vice-versa
char_to_label = {char: idx for idx, char in enumerate(CHARACTERS)}
label_to_char = {idx: char for idx, char in enumerate(CHARACTERS)}

class Captcha(object):
    def __init__(self):
        if not TENSORFLOW_AVAILABLE:
            self.model = None
            print("Warning: TensorFlow not found. Model cannot be built.")
            return
        self.model = self._build_model()

    def _build_model(self):
        """Builds a character recognition model using MobileNetV2 as a base."""
        # Input layer for character images (resized and 3-channel for MobileNetV2)
        input_tensor = Input(shape=(CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, 3))

        # Load MobileNetV2 pre-trained on ImageNet, without the top classification layer
        base_model = MobileNetV2(
            input_tensor=input_tensor,
            weights='imagenet',     # Load weights pre-trained on ImageNet
            include_top=False,      # Exclude the original ImageNet classifier (1000 classes)
            alpha=1.0               # Alpha controls the width of the network. 1.0 is standard.
        )

        # Freeze the layers of the base model
        # We don't want to update their weights during initial training on our small dataset
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x) # Global Average Pooling reduces spatial dimensions
        # Optional: Add an intermediate dense layer if needed, but for small data, simpler is often better
        # x = Dense(128, activation='relu')(x)
        # x = BatchNormalization()(x) # Good to use with Dense layers
        # x = Dropout(0.5)(x) # Helps prevent overfitting
        output_tensor = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer for our characters

        model = Model(inputs=input_tensor, outputs=output_tensor)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam is a good default optimizer
                      loss='categorical_crossentropy',    # Suitable for multi-class classification
                      metrics=['accuracy'])
        return model

    def _preprocess_char_image(self, char_img_pil):
        """Preprocesses a single character PIL Image for the transfer learning model."""
        # Convert to RGB (MobileNetV2 expects 3 channels)
        img = char_img_pil.convert('RGB')
        # Resize to the input size expected by MobileNetV2
        img = img.resize((CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)
        # Expand dims to create a batch of 1 (model expects batch dimension)
        img_array_batch = np.expand_dims(img_array, axis=0)
        # Preprocess input specifically for MobileNetV2 (scales pixel values, e.g., to [-1, 1])
        processed_img_array = mobilenet_preprocess_input(img_array_batch)

        return processed_img_array[0] # Return the single preprocessed image array (remove batch dim)

    def _segment_captcha_image(self, full_img_pil):
        """Segments a full CAPTCHA PIL Image into a list of character PIL Images."""
        char_pil_images = []
        width, height = full_img_pil.size
        # Assuming characters are roughly evenly spaced and non-overlapping.
        char_segment_width = width // NUM_CHARACTERS_PER_CAPTCHA

        for i in range(NUM_CHARACTERS_PER_CAPTCHA):
            left = i * char_segment_width
            top = 0
            # Ensure right boundary doesn't exceed image width
            right = min((i + 1) * char_segment_width, width)
            bottom = height
            
            if right - left < 5: # Sanity check for very narrow segments
                print(f"Warning: Character segment {i} in image is too narrow ({right-left}px). Appending a black placeholder.")
                char_pil_images.append(Image.new('RGB', (CHAR_IMG_WIDTH, CHAR_IMG_HEIGHT), 'black'))
                continue

            char_img_pil = full_img_pil.crop((left, top, right, bottom))
            char_pil_images.append(char_img_pil)
        return char_pil_images

    def load_and_prepare_data_from_samples(self, captcha_samples_with_labels):
        """
        Loads CAPTCHA images, segments them into characters, preprocesses characters,
        and prepares them for training.
        Args:
            captcha_samples_with_labels: A list of tuples (image_path, "LABELSTRING").
        Returns:
            (X, y): Tuple of (character_images_array, character_labels_array).
        """
        char_images_data = []
        char_labels_data = []

        for img_path, label_string in captcha_samples_with_labels:
            if not os.path.exists(img_path):
                print(f"Warning: Image path {img_path} does not exist. Skipping.")
                continue
            if len(label_string) != NUM_CHARACTERS_PER_CAPTCHA:
                print(f"Warning: Label '{label_string}' for {img_path} does not have {NUM_CHARACTERS_PER_CAPTCHA} chars. Skipping.")
                continue

            try:
                full_img_pil = Image.open(img_path)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}. Skipping.")
                continue
                
            character_pil_segments = self._segment_captcha_image(full_img_pil)

            if len(character_pil_segments) != NUM_CHARACTERS_PER_CAPTCHA:
                print(f"Warning: Segmentation of {img_path} resulted in {len(character_pil_segments)} segments, expected {NUM_CHARACTERS_PER_CAPTCHA}. Skipping this CAPTCHA.")
                continue

            for i in range(NUM_CHARACTERS_PER_CAPTCHA):
                char_label_char = label_string[i] # The actual character e.g. 'A'
                if char_label_char not in char_to_label:
                    print(f"Warning: Character '{char_label_char}' in label '{label_string}' for {img_path} is not in defined CHARACTERS. Skipping this character.")
                    continue

                char_img_pil_segment = character_pil_segments[i]
                processed_char_img = self._preprocess_char_image(char_img_pil_segment)
                char_images_data.append(processed_char_img)
                char_labels_data.append(char_to_label[char_label_char]) # Append integer label

        if not char_images_data: # If no images were successfully processed
            return np.array([]), np.array([])
        return np.array(char_images_data), np.array(char_labels_data)


    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=8):
        """Trains the model (only the new head if base_model.trainable=False)."""
        if self.model is None:
            print("Error: Model not built (TensorFlow might be missing or failed to initialize). Cannot train.")
            return None
        if len(X_train) == 0 or len(X_val) == 0:
            print("Error: Training or validation set for characters is empty. Cannot train.")
            return None

        # One-hot encode labels for categorical_crossentropy
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)

        # Callbacks for better training
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

        print(f"Starting training with {len(X_train)} training character samples and {len(X_val)} validation character samples.")
        history = self.model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping, reduce_lr]
        )
        print("Training complete.")
        return history

    def __call__(self, im_path, save_path):
        """
        Algo for inference on a single CAPTCHA image.
        Args:
            im_path: .jpg image path to load and to infer.
            save_path: output file path to save the one-line outcome.
        """
        if self.model is None:
            print("Error: Model not built or not trained. Cannot predict.")
            if save_path:
                with open(save_path, 'w') as f: f.write("ERROR_MODEL_NOT_AVAILABLE")
            return "ERROR_MODEL_NOT_AVAILABLE"
        try:
            full_img_pil = Image.open(im_path)
            character_pil_segments = self._segment_captcha_image(full_img_pil)
            
            if len(character_pil_segments) != NUM_CHARACTERS_PER_CAPTCHA:
                 msg = f"ERROR_SEGMENTATION_WRONG_COUNT_{len(character_pil_segments)}"
                 print(f"Error: Segmentation of {im_path} resulted in {len(character_pil_segments)} segments. Expected {NUM_CHARACTERS_PER_CAPTCHA}")
                 if save_path:
                    with open(save_path, 'w') as f: f.write(msg)
                 return msg

            predicted_text = ""
            for char_img_pil_segment in character_pil_segments:
                processed_char_img_single = self._preprocess_char_image(char_img_pil_segment)
                # Add batch dimension for prediction: (1, height, width, channels)
                processed_char_img_batch = np.expand_dims(processed_char_img_single, axis=0)

                prediction = self.model.predict(processed_char_img_batch, verbose=0) # verbose=0 for cleaner output
                predicted_label_idx = np.argmax(prediction, axis=1)[0]
                predicted_text += label_to_char[predicted_label_idx]

            with open(save_path, 'w') as f:
                f.write(predicted_text)
            return predicted_text
        except FileNotFoundError:
            print(f"Error: Image file not found at {im_path}")
            if save_path:
                with open(save_path, 'w') as f: f.write("ERROR_IMAGE_NOT_FOUND")
            return "ERROR_IMAGE_NOT_FOUND"
        except Exception as e:
            print(f"Error processing {im_path} for prediction: {e} (Type: {type(e).__name__})")
            if save_path:
                with open(save_path, 'w') as f: f.write(f"ERROR_PROCESSING_{type(e).__name__}")
            return f"ERROR_PROCESSING_{type(e).__name__}"

    def evaluate_model(self, X_test, y_test):
        if self.model is None or len(X_test) == 0 or len(y_test) == 0:
            print("Error: Model not built or test set for characters is empty. Cannot evaluate.")
            return None, None
        y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)
        loss, accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test Set - Loss: {loss:.4f}, Character Accuracy: {accuracy:.4f}")
        return loss, accuracy

# --- Main execution block: Handles data loading, training, and demonstration ---
if __name__ == '__main__':
    if not TENSORFLOW_AVAILABLE:
        print("Exiting: TensorFlow is required for this script to run fully but is not installed.")
    else:
        print(f"TensorFlow version: {tf.__version__}")

        # --- 1. Define data paths based on your specified structure ---
        BASE_SAMPLE_DIR = "sampleCaptchas" # Main folder containing 'input' and 'output' subdirs
        INPUT_IMAGE_SUBDIR = "input"
        OUTPUT_LABEL_SUBDIR = "output"

        # Construct full paths
        input_captcha_dir = os.path.join(BASE_SAMPLE_DIR, INPUT_IMAGE_SUBDIR)
        output_label_dir = os.path.join(BASE_SAMPLE_DIR, OUTPUT_LABEL_SUBDIR)

        # --- 2. Discover samples and their labels dynamically ---
        def discover_captcha_samples(image_dir, label_dir):
            discovered_samples = []
            # Loop for filenames input00.jpg to input24.jpg
            for i in range(25):
                idx_str = f"{i:02d}" # Formats number as two digits, e.g., 0, 1 -> 00, 01
                img_filename = f"input{idx_str}.jpg"
                label_filename = f"output{idx_str}.txt"

                current_img_path = os.path.join(image_dir, img_filename)
                current_label_path = os.path.join(label_dir, label_filename)

                # Check if both image and label file exist
                if os.path.exists(current_img_path) and os.path.exists(current_label_path):
                    try:
                        with open(current_label_path, 'r') as f:
                            label_text = f.read().strip().upper() # Read label, remove whitespace, ensure uppercase
                        
                        # Validate label
                        if len(label_text) == NUM_CHARACTERS_PER_CAPTCHA and all(c in CHARACTERS for c in label_text):
                            discovered_samples.append((current_img_path, label_text))
                        else:
                            if len(label_text) != NUM_CHARACTERS_PER_CAPTCHA:
                                print(f"Warning: Label in {current_label_path} ('{label_text}') is not {NUM_CHARACTERS_PER_CAPTCHA} characters long. Skipping this sample.")
                            else: # Contains invalid characters
                                print(f"Warning: Label in {current_label_path} ('{label_text}') contains characters not in CHARACTERS list. Skipping this sample.")
                    except Exception as e:
                        print(f"Error reading or processing label file {current_label_path}: {e}. Skipping this sample.")
                else:
                    # Optionally print if files are missing, useful for debugging setup
                    if not os.path.exists(current_img_path):
                        print(f"Info: Image file {current_img_path} not found during discovery. Will be skipped.")
                    if not os.path.exists(current_label_path):
                         print(f"Info: Label file {current_label_path} not found during discovery. Will be skipped.")
            return discovered_samples

        all_samples_data = discover_captcha_samples(input_captcha_dir, output_label_dir)

        if not all_samples_data:
            print("\nCRITICAL: No valid CAPTCHA samples found. Please ensure the following structure and content:")
            print(f"1. A base directory named '{BASE_SAMPLE_DIR}' exists in the same location as this script.")
            print(f"2. Inside '{BASE_SAMPLE_DIR}', an '{INPUT_IMAGE_SUBDIR}' directory exists.")
            print(f"3. Inside '{INPUT_IMAGE_SUBDIR}', you have 'input00.jpg' through 'input24.jpg'.")
            print(f"4. Inside '{BASE_SAMPLE_DIR}', an '{OUTPUT_LABEL_SUBDIR}' directory exists.")
            print(f"5. Inside '{OUTPUT_LABEL_SUBDIR}', you have 'output00.txt' through 'output24.txt'.")
            print(f"6. Each 'outputXX.txt' file contains exactly {NUM_CHARACTERS_PER_CAPTCHA} uppercase alphanumeric characters ({CHARACTERS}).")
            print("Exiting due to no data.")
        else:
            print(f"Discovered {len(all_samples_data)} valid CAPTCHA samples to process.")

            # --- 3. Instantiate the Captcha solver ---
            captcha_solver = Captcha()

            if captcha_solver.model is None: # Check if model building failed (e.g., TF not available)
                 print("Exiting: Captcha model could not be initialized.")
            else:
                # --- 4. Load and prepare all character data from the discovered CAPTCHAs ---
                print("Loading and preparing character data from all samples...")
                X_all_chars, y_all_chars = captcha_solver.load_and_prepare_data_from_samples(all_samples_data)
                
                if X_all_chars.size == 0 or y_all_chars.size == 0:
                    print("CRITICAL: No character data was loaded from the samples. This might be due to issues opening images, segmenting characters, or invalid labels. Please check warnings above. Exiting.")
                else:
                    print(f"Successfully loaded {len(X_all_chars)} individual character images and {len(y_all_chars)} corresponding labels.")

                    # --- 5. Split character data into training, validation, and test sets ---
                    # Simple split: 60% train, 20% validation, 20% test
                    # First split: 60% train, 40% remaining
                    X_train_chars, X_temp_val_test, y_train_chars, y_temp_val_test = train_test_split(
                        X_all_chars, y_all_chars, test_size=0.4, random_state=42
                    )
                    
                    # Second split: Split the remaining 40% into equal validation and test sets
                    X_val_chars, X_test_chars, y_val_chars, y_test_chars = train_test_split(
                        X_temp_val_test, y_temp_val_test, test_size=0.5, random_state=42
                    )

                    print(f"Training characters: {len(X_train_chars)}")
                    print(f"Validation characters: {len(X_val_chars)}")
                    print(f"Test characters: {len(X_test_chars)}")

                    # --- 6. Train the model ---
                    if len(X_train_chars) > 0 and len(X_val_chars) > 0:
                        print("Starting model training...")
                        captcha_solver.train(X_train_chars, y_train_chars, X_val_chars, y_val_chars, epochs=30, batch_size=8)
                    else:
                        print("Training or validation set for characters is empty. Skipping training. This can occur if too few valid samples were found or due to the split of a very small dataset.")

                    # --- 7. Evaluate on the test set ---
                    print("\n=== Model Evaluation ===")
                    
                    # First evaluate character-level accuracy
                    if len(X_test_chars) > 0 and len(y_test_chars) > 0:
                        print("\n--- Character-Level Evaluation ---")
                        loss, char_accuracy = captcha_solver.evaluate_model(X_test_chars, y_test_chars)
                    else:
                        print("Test set for characters is empty, skipping character-level evaluation.")
                    
                    # Now evaluate full CAPTCHA accuracy on test samples
                    print("\n--- Full CAPTCHA Evaluation ---")
                    test_results = []
                    total_correct_chars = 0
                    total_chars = 0
                    
                    # Get test set samples (approximately 20% of all samples)
                    num_test_samples = max(1, len(all_samples_data) // 5)  # At least 1 test sample
                    test_samples = all_samples_data[-num_test_samples:]  # Take last 20% as test set
                    
                    print(f"Testing on {len(test_samples)} full CAPTCHA images...")
                    
                    for test_img_path, true_label in test_samples:
                        # Create output filename for this test sample
                        base_name = os.path.basename(test_img_path)
                        name_part = os.path.splitext(base_name)[0]
                        test_output_filename = f"predicted_{name_part}.txt"
                        
                        # Get prediction
                        predicted_text = captcha_solver(test_img_path, test_output_filename)
                        
                        # Calculate character-level accuracy for this sample
                        correct_chars = sum(1 for p, t in zip(predicted_text, true_label) if p == t)
                        total_correct_chars += correct_chars
                        total_chars += len(true_label)
                        
                        # Store results
                        is_correct = predicted_text == true_label
                        test_results.append({
                            'image': base_name,
                            'predicted': predicted_text,
                            'actual': true_label,
                            'is_correct': is_correct,
                            'correct_chars': correct_chars,
                            'total_chars': len(true_label)
                        })
                        
                        # Print individual result
                        print(f"\nTest image: {base_name}")
                        print(f"Predicted: {predicted_text}")
                        print(f"Actual:    {true_label}")
                        print(f"Correct:   {'✓' if is_correct else '✗'}")
                        print(f"Character accuracy: {correct_chars}/{len(true_label)} ({correct_chars/len(true_label)*100:.1f}%)")
                    
                    # Calculate and print overall statistics
                    num_correct = sum(1 for r in test_results if r['is_correct'])
                    captcha_accuracy = num_correct / len(test_results) if test_results else 0
                    char_level_accuracy = total_correct_chars / total_chars if total_chars > 0 else 0
                    
                    print("\n=== Final Test Results ===")
                    print(f"Total test samples: {len(test_results)}")
                    print(f"Fully correct CAPTCHAs: {num_correct}/{len(test_results)} ({captcha_accuracy*100:.1f}%)")
                    print(f"Character-level accuracy: {total_correct_chars}/{total_chars} ({char_level_accuracy*100:.1f}%)")
                    
                    # Save test results to a file
                    with open("test_results.txt", "w") as f:
                        f.write("=== CAPTCHA Solver Test Results ===\n\n")
                        f.write(f"Total test samples: {len(test_results)}\n")
                        f.write(f"Fully correct CAPTCHAs: {num_correct}/{len(test_results)} ({captcha_accuracy*100:.1f}%)\n")
                        f.write(f"Character-level accuracy: {total_correct_chars}/{total_chars} ({char_level_accuracy*100:.1f}%)\n\n")
                        f.write("Detailed Results:\n")
                        for r in test_results:
                            f.write(f"\nImage: {r['image']}\n")
                            f.write(f"Predicted: {r['predicted']}\n")
                            f.write(f"Actual:    {r['actual']}\n")
                            f.write(f"Correct:   {'Yes' if r['is_correct'] else 'No'}\n")
                            f.write(f"Character accuracy: {r['correct_chars']}/{r['total_chars']}\n")

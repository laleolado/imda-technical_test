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
        # Simpler head for potentially better generalization with small data
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x) # Good to use with Dense layers
        x = Dropout(0.5)(x) # Helps prevent overfitting
        
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

    def _augment_batch(self, images):
        """Apply random augmentations to a batch of images"""
        augmented = []
        
        # Initialize augmentation layers once
        rotation = tf.keras.layers.RandomRotation(factor=0.1)  # ±10 degrees = ±0.1 radians
        translation = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
        zoom = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
        
        for image in images:
            # Convert to tensor and add batch dimension
            img = tf.convert_to_tensor(image)
            img = tf.cast(img, tf.float32)
            img = tf.expand_dims(img, 0)  # Add batch dimension
            
            # Random rotation
            if tf.random.uniform([]) > 0.5:
                img = rotation(img)
            
            # Random translation
            if tf.random.uniform([]) > 0.5:
                img = translation(img)
            
            # Random zoom
            if tf.random.uniform([]) > 0.5:
                img = zoom(img)
            
            # Random brightness
            if tf.random.uniform([]) > 0.5:
                img = tf.image.random_brightness(img, 0.2)
            
            # Remove batch dimension and ensure values are in valid range
            img = tf.squeeze(img, axis=0)
            img = tf.clip_by_value(img, -1.0, 1.0)  # For MobileNetV2 preprocessed inputs
            augmented.append(img.numpy())
            
        return np.array(augmented)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Trains the model with custom data augmentation."""
        if self.model is None:
            print("Error: Model not built (TensorFlow might be missing or failed to initialize). Cannot train.")
            return None
        if len(X_train) == 0 or len(X_val) == 0:
            print("Error: Training or validation set for characters is empty. Cannot train.")
            return None

        # One-hot encode labels for categorical_crossentropy
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)

        print(f"Starting training with {len(X_train)} training character samples and {len(X_val)} validation character samples.")
        
        # Custom training loop with data augmentation
        steps_per_epoch = max(len(X_train) * 4 // batch_size, 1)  # 4x augmentation
        
        # Initialize variables for early stopping and learning rate reduction
        best_val_loss = float('inf')
        patience = 15
        min_delta = 0.001
        wait = 0
        
        # Learning rate reduction parameters
        lr_patience = 7
        lr_wait = 0
        lr_factor = 0.5
        min_lr = 0.00001
        
        # Define the training step function with tf.function and input signatures
        @tf.function(reduce_retracing=True,
                    input_signature=[
                        tf.TensorSpec(shape=(None, CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
                    ])
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, predictions

        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Shuffle indices for this epoch
            indices = np.random.permutation(len(X_train))
            epoch_loss = []
            epoch_accuracy = []
            
            for step in range(steps_per_epoch):
                # Get batch indices with wrapping
                start_idx = (step * batch_size) % len(X_train)
                end_idx = min(start_idx + batch_size, len(X_train))
                if end_idx - start_idx < batch_size:  # Wrap around
                    batch_indices = np.concatenate([
                        indices[start_idx:end_idx],
                        indices[:batch_size - (end_idx - start_idx)]
                    ])
                else:
                    batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_x = X_train[batch_indices]
                batch_y = y_train_cat[batch_indices]
                
                # Apply augmentation
                batch_x = self._augment_batch(batch_x)
                
                # Convert to tensors with explicit shapes
                batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
                batch_x = tf.ensure_shape(batch_x, [None, CHAR_IMG_HEIGHT, CHAR_IMG_WIDTH, 3])
                batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)
                batch_y = tf.ensure_shape(batch_y, [None, NUM_CLASSES])
                
                # Train step
                loss, predictions = train_step(batch_x, batch_y)
                epoch_loss.append(loss.numpy())
                epoch_accuracy.append(tf.reduce_mean(tf.keras.metrics.categorical_accuracy(batch_y, predictions)).numpy())
            
            # Calculate average metrics for this epoch
            avg_loss = np.mean(epoch_loss)
            avg_accuracy = np.mean(epoch_accuracy)
            
            # Evaluate on validation set
            val_metrics = self.model.evaluate(X_val, y_val_cat, verbose=0)
            val_loss = val_metrics[0]
            val_accuracy = val_metrics[1]
            
            # Store metrics in history
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0
                # Save best weights
                self.model.save_weights('best_model.weights.h5')
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    # Restore best weights
                    self.model.load_weights('best_model.weights.h5')
                    break
            
            # Learning rate reduction
            if val_loss < best_val_loss - min_delta:
                lr_wait = 0
            else:
                lr_wait += 1
                if lr_wait >= lr_patience:
                    # Get current learning rate
                    current_lr = self.model.optimizer.learning_rate
                    if isinstance(current_lr, tf.Variable):
                        current_lr = current_lr.numpy()
                    current_lr = float(current_lr)
                    
                    if current_lr > min_lr:
                        new_lr = max(current_lr * lr_factor, min_lr)
                        # Update learning rate using assign
                        self.model.optimizer.learning_rate.assign(new_lr)
                        print(f"Reducing learning rate to {new_lr}")
                        lr_wait = 0
        
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

            # --- 3. Split samples into train, validation, and test sets ---
            # Split samples before extracting characters (60% train, 20% validation, 20% test)
            train_samples, temp_val_test = train_test_split(all_samples_data, test_size=0.4, random_state=42)
            val_samples, test_samples = train_test_split(temp_val_test, test_size=0.5, random_state=42)

            print(f"\nSplit samples into:")
            print(f"Training samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")
            print(f"Test samples: {len(test_samples)}")

            # --- 4. Instantiate the Captcha solver ---
            captcha_solver = Captcha()

            if captcha_solver.model is None:
                print("Exiting: Captcha model could not be initialized.")
            else:
                # --- 5. Load and prepare character data from training and validation samples ---
                print("\nLoading and preparing character data from training samples...")
                X_train_chars, y_train_chars = captcha_solver.load_and_prepare_data_from_samples(train_samples)
                
                print("Loading and preparing character data from validation samples...")
                X_val_chars, y_val_chars = captcha_solver.load_and_prepare_data_from_samples(val_samples)
                
                if X_train_chars.size == 0 or y_train_chars.size == 0:
                    print("CRITICAL: No character data was loaded from training samples. Exiting.")
                else:
                    print(f"\nPrepared data statistics:")
                    print(f"Training characters: {len(X_train_chars)}")
                    print(f"Validation characters: {len(X_val_chars)}")

                    # --- 6. Train the model ---
                    if len(X_train_chars) > 0 and len(X_val_chars) > 0:
                        print("\nStarting model training...")
                        history = captcha_solver.train(X_train_chars, y_train_chars, X_val_chars, y_val_chars, epochs=50, batch_size=16)
                        if history:
                            print("\nTraining history:")
                            for metric, values in history.items():
                                print(f"{metric}: {values[-1]:.4f}")
                    else:
                        print("Training or validation set for characters is empty. Skipping training.")

                    # --- 7. Test the model on test samples only ---
                    if test_samples:
                        print("\n--- Testing Model on Test Set Only ---")
                        correct_predictions = 0
                        total_predictions = 0
                        
                        # Create a directory for test predictions if it doesn't exist
                        test_output_dir = "test_predictions"
                        if not os.path.exists(test_output_dir):
                            os.makedirs(test_output_dir)
                            
                        # Test only on test set samples
                        for img_path, actual_label in test_samples:
                            base_name = os.path.splitext(os.path.basename(img_path))[0]
                            output_path = os.path.join(test_output_dir, f"predicted_{base_name}.txt")
                            
                            # Make prediction
                            predicted_text = captcha_solver(img_path, output_path)
                            
                            # Print results
                            print(f"\nTesting {base_name}:")
                            print(f"Predicted: {predicted_text}")
                            print(f"Actual:    {actual_label}")
                            
                            if predicted_text == actual_label:
                                correct_predictions += 1
                                print("✓ Correct!")
                            else:
                                char_matches = sum(1 for p, a in zip(predicted_text, actual_label) if p == a)
                                char_accuracy = char_matches / len(actual_label) if len(predicted_text) == len(actual_label) else 0
                                print(f"✗ Incorrect (Character-level accuracy: {char_accuracy:.2%})")
                            
                            total_predictions += 1
                        
                        # Print final statistics
                        print("\n--- Final Test Set Results ---")
                        accuracy = correct_predictions / total_predictions
                        print(f"Test Set Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct)")
                        print(f"All test predictions have been saved in the '{test_output_dir}' directory")
                    else:
                        print("No test samples available, skipping testing.")

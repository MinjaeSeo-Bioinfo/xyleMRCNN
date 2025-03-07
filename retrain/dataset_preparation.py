import os
import glob
import json
import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import shutil

class PredictionReviewer:
    def __init__(self, images_dir, predictions_json, filename_prefix="annotated_", output_dir=None):
        """
        Class for reviewing and filtering prediction results
        
        Args:
            images_dir: Directory containing image files
            predictions_json: Path to JSON file containing prediction results
            filename_prefix: Image filename prefix (default: "annotated_")
            output_dir: Directory to save output files (default: current directory)
        """
        self.images_dir = images_dir
        self.predictions_json = predictions_json
        self.filename_prefix = filename_prefix
        self.output_dir = output_dir or os.getcwd()  # Default is current directory
        
        # Load predictions JSON
        with open(predictions_json, 'r') as f:
            self.predictions_data = json.load(f)
        
        # Image information mapping
        self.image_info = {}
        for img in self.predictions_data.get('images', []):
            self.image_info[img['id']] = img
        
        # Annotation information mapping (by image ID)
        self.image_annotations = {}
        for ann in self.predictions_data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Create actual filename mapping
        self.filename_mapping = {}
        self.reverse_mapping = {}
        for img_id, img_data in self.image_info.items():
            original_name = img_data['file_name']
            prefixed_name = f"{filename_prefix}{original_name}"
            self.filename_mapping[original_name] = prefixed_name
            self.reverse_mapping[prefixed_name] = original_name
        
        # List of all image IDs
        self.image_ids = sorted(list(self.image_info.keys()))
        
        # Set of rejected annotation IDs
        self.rejected_ids = set()
        
        # Current state
        self.current_img_idx = 0
        self.current_ann_idx = 0
        
        # Initialize with first image ID
        self.current_img_id = self.image_ids[0] if self.image_ids else None
        
        print(f"Loaded information for {len(self.image_ids)} images.")
        print(f"Total of {sum(len(anns) for anns in self.image_annotations.values())} annotations.")
        print(f"Output files will be saved to {self.output_dir}.")
        
        # Initialize UI
        self.setup_widgets()
    
    def setup_widgets(self):
        # Image selection dropdown (displayed with actual filenames)
        img_options = [(f"{i}: {self.image_info[img_id]['file_name']}", img_id) 
                       for i, img_id in enumerate(self.image_ids)]
        
        self.image_dropdown = widgets.Dropdown(
            options=img_options,
            description='Image:',
            style={'description_width': 'initial'},
            value=self.image_ids[0]  # Set first image as default
        )
        self.image_dropdown.observe(self.on_image_change, names='value')
        
        # Create buttons (same as existing code)
        self.reject_button = widgets.Button(
            description='Reject',
            button_style='danger',
            tooltip='Reject this annotation'
        )
        
        self.accept_button = widgets.Button(
            description='Accept',
            button_style='success',
            tooltip='Accept this annotation'
        )
        
        self.prev_button = widgets.Button(
            description='Previous',
            tooltip='Go to previous annotation'
        )
        
        self.next_button = widgets.Button(
            description='Next',
            tooltip='Go to next annotation'
        )
        
        self.save_button = widgets.Button(
            description='Save Filtered Predictions',
            button_style='info',
            tooltip='Save filtered predictions'
        )
        
        # Progress indicator
        self.progress = widgets.IntProgress(
            value=0,
            min=0,
            max=1,  # Will be updated later
            description='Progress:',
            bar_style='info'
        )
        
        # Status label
        self.status_label = widgets.Label(value='Ready. Please select an image.')
        
        # Connect events
        self.reject_button.on_click(self.reject)
        self.accept_button.on_click(self.accept)
        self.prev_button.on_click(self.prev)
        self.next_button.on_click(self.next)
        self.save_button.on_click(self.save_filtered_predictions)
        
        # Widget layout
        self.nav_buttons = widgets.HBox([self.prev_button, self.next_button])
        self.action_buttons = widgets.HBox([self.reject_button, self.accept_button, self.save_button])
        self.controls = widgets.VBox([
            self.image_dropdown,
            self.progress,
            self.nav_buttons,
            self.action_buttons,
            self.status_label
        ])
        
        # Output area
        self.output = widgets.Output()
        
        # Load first image and display annotation
        if self.image_ids:
            # Force load first image
            self.current_img_id = self.image_ids[0]
            self.current_ann_idx = 0
            self.load_current_image()
            self.show_current_annotation()
        
        # Display widgets
        display(self.controls)
        display(self.output)
    
    def on_image_change(self, change):
        """Function called when image dropdown changes"""
        if change['type'] == 'change' and change['name'] == 'value':
            self.current_img_id = change['new']
            self.current_ann_idx = 0
            self.load_current_image()
            self.show_current_annotation()
    
    def load_current_image(self):
        """Load currently selected image"""
        if not self.current_img_id:
            return
        
        img_data = self.image_info[self.current_img_id]
        original_filename = img_data['file_name']
        prefixed_filename = self.filename_mapping[original_filename]
        
        image_path = os.path.join(self.images_dir, prefixed_filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            # Try filename without prefix
            image_path = os.path.join(self.images_dir, original_filename)
            if not os.path.exists(image_path):
                self.status_label.value = f"Image file not found: {prefixed_filename} or {original_filename}"
                self.image = None
                return
        
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            self.status_label.value = f"Unable to load image: {image_path}"
            self.image = None
            return
        
        # Update number of annotations for current image
        current_anns = self.image_annotations.get(self.current_img_id, [])
        self.progress.max = len(current_anns) - 1 if len(current_anns) > 0 else 0
        
        self.status_label.value = f"Image loaded: {os.path.basename(image_path)}"
    
    def show_current_annotation(self):
        """Display current annotation"""
        with self.output:
            clear_output(wait=True)
            
            if self.image is None:
                print("Unable to load image.")
                return
            
            # List of annotations for current image
            current_anns = self.image_annotations.get(self.current_img_id, [])
            
            if not current_anns:
                print(f"No annotations for image ID {self.current_img_id}.")
                plt.figure(figsize=(12, 8))
                plt.imshow(self.image)
                plt.title("No annotations")
                plt.axis('off')
                plt.show()
                return
            
            # Copy image
            img_copy = self.image.copy()
            
            # Current annotation
            if self.current_ann_idx < len(current_anns):
                ann = current_anns[self.current_ann_idx]
                
                # Extract bounding box (convert COCO format [x, y, width, height] to [x1, y1, x2, y2])
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = map(float, bbox)
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    
                    # Draw bounding box
                    ann_id = ann.get('id')
                    color = (255, 0, 0) if ann_id in self.rejected_ids else (0, 255, 0)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # Text information
                    status = "REJECTED" if ann_id in self.rejected_ids else "ACCEPTED"
                    class_id = ann.get('category_id', 0)
                    score = ann.get('score', 1.0)
                    
                    label = f"Class: {class_id}, ID: {ann_id}, Score: {score:.2f}, Status: {status}"
                    cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Update progress
                    self.progress.value = self.current_ann_idx
                    
                    # plt.figure(figsize=(10, 8))
                    
                    # Add padding (include area around bounding box)
                    padding = max(30, int(max(w, h) * 0.3))
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(img_copy.shape[1], x2 + padding)
                    crop_y2 = min(img_copy.shape[0], y2 + padding)
                                        
                    # Crop zoomed area
                    zoomed = img_copy[crop_y1:crop_y2, crop_x1:crop_x2]
                                        
                    # Image title (including zoomed annotation info)
                    img_filename = self.image_info[self.current_img_id]['file_name']
                    plt.title(f"{img_filename} - Annotation {self.current_ann_idx + 1}/{len(current_anns)}\n{label}")
                                        
                    plt.imshow(zoomed)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Invalid bounding box format: {bbox}")
            else:
                print(f"Invalid annotation index: {self.current_ann_idx}")
    
    def reject(self, b):
        """Reject current annotation"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns):
            ann = current_anns[self.current_ann_idx]
            ann_id = ann.get('id')
            
            if ann_id is not None:
                if ann_id in self.rejected_ids:
                    self.rejected_ids.remove(ann_id)
                    self.status_label.value = f"Removed ID {ann_id} from rejected list."
                else:
                    self.rejected_ids.add(ann_id)
                    self.status_label.value = f"Added ID {ann_id} to rejected list."
                
                self.show_current_annotation()
                self.next(None)  # Automatically move to next annotation
    
    def accept(self, b):
        """Accept current annotation"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns):
            ann = current_anns[self.current_ann_idx]
            ann_id = ann.get('id')
            
            if ann_id is not None and ann_id in self.rejected_ids:
                self.rejected_ids.remove(ann_id)
                self.status_label.value = f"Accepted ID {ann_id}."
                self.show_current_annotation()
            
            self.next(None)  # Automatically move to next annotation
    
    def prev(self, b):
        """Move to previous annotation"""
        if self.current_ann_idx > 0:
            self.current_ann_idx -= 1
            self.show_current_annotation()
    
    def next(self, b):
        """Move to next annotation"""
        current_anns = self.image_annotations.get(self.current_img_id, [])
        
        if self.current_ann_idx < len(current_anns) - 1:
            self.current_ann_idx += 1
            self.show_current_annotation()
    
    def save_filtered_predictions(self, b):
        """Save filtered predictions"""
        # Save rejected ID list (for future reference)
        rejected_path = os.path.join(self.output_dir, 'rejected_annotations.txt')
        with open(rejected_path, 'w') as f:
            for ann_id in sorted(self.rejected_ids):
                f.write(f"{ann_id}\n")
        
        # Create filtered annotation list (excluding rejected IDs)
        filtered_annotations = []
        for annotations in self.image_annotations.values():
            for ann in annotations:
                if ann.get('id') not in self.rejected_ids:
                    filtered_annotations.append(ann)
        
        # Reassign annotation IDs (to consecutive numbers)
        for i, ann in enumerate(filtered_annotations, 1):
            ann['id'] = i
        
        # Create filtered prediction data
        filtered_predictions = {
            'images': list(self.image_info.values()),
            'annotations': filtered_annotations,
            'categories': self.predictions_data.get('categories', [])
        }
        
        # Reassign image IDs to consecutive numbers
        for i, img in enumerate(filtered_predictions['images'], 1):
            old_id = img['id']
            img['id'] = i
            
            # Update image_id in annotations for this image
            for ann in filtered_annotations:
                if ann['image_id'] == old_id:
                    ann['image_id'] = i
        
        # Save filtered predictions
        output_json = os.path.join(self.output_dir, 'filtered_predictions.json')
        with open(output_json, 'w') as f:
            json.dump(filtered_predictions, f, indent=2)
        
        # Save image-to-rejected-IDs information (for debugging)
        rejected_ids_json = os.path.join(self.output_dir, 'image_to_rejected_ids.json')
        image_to_rejected = {}
        for img_id in self.image_ids:
            img_filename = self.image_info[img_id]['file_name']
            anns = self.image_annotations.get(img_id, [])
            rejected_in_img = []
            
            for ann in anns:
                ann_id = ann.get('id')
                if ann_id in self.rejected_ids:
                    rejected_in_img.append(ann_id)
            
            if rejected_in_img:
                image_to_rejected[img_filename] = rejected_in_img
        
        with open(rejected_ids_json, 'w') as f:
            json.dump(image_to_rejected, f, indent=2)
        
        # Prepare training dataset
        try:
            train_dir = self.prepare_training_dataset()
            success_msg = f"Training dataset prepared in {train_dir} directory."
        except Exception as e:
            success_msg = f"Error while preparing training dataset: {str(e)}"
        
        self.status_label.value = f"Filtered predictions saved to {output_json}. {success_msg}"
        with self.output:
            clear_output(wait=True)
            print(f"Total of {len(self.rejected_ids)} annotations were rejected.")
            print(f"Number of filtered annotations: {len(filtered_annotations)}")
            print(f"Filtered predictions saved to {output_json}.")
            print(success_msg)
    
    def prepare_training_dataset(self):
        """Prepare dataset for retraining (filtered predictions + images)"""
        # Create training directory
        train_dir = os.path.join(self.output_dir, "retrain_dataset")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create annotations directory
        ann_dir = os.path.join(train_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        # Create images directory
        img_dir = os.path.join(train_dir, "train")
        os.makedirs(img_dir, exist_ok=True)
        
        # Save filtered predictions as annotation file
        filtered_json_path = os.path.join(ann_dir, "result_train.json")
        filtered_predictions_path = os.path.join(self.output_dir, "filtered_predictions.json")
        shutil.copy(filtered_predictions_path, filtered_json_path)
        
        # Copy images
        for img_id, img_info in self.image_info.items():
            original_name = img_info['file_name']
            prefixed_name = self.filename_mapping.get(original_name, original_name)
            
            # Image file path
            src_path = os.path.join(self.images_dir, prefixed_name)
            if not os.path.exists(src_path):
                src_path = os.path.join(self.images_dir, original_name)
            
            if os.path.exists(src_path):
                # Copy to training directory (without prefix)
                dst_path = os.path.join(img_dir, original_name)
                shutil.copy(src_path, dst_path)
            else:
                print(f"Image not found: {prefixed_name} or {original_name}")
        
        print(f"Training dataset prepared in {train_dir} directory.")
        print(f"Number of images: {len(self.image_info)}")
        
        return train_dir

# Usage
def start_prediction_review(images_dir, predictions_json, filename_prefix="annotated_", output_dir=None):
    """
    Start prediction review
    
    Args:
        images_dir: Directory containing image files
        predictions_json: Path to JSON file containing prediction results
        filename_prefix: Image filename prefix (default: "annotated_")
        output_dir: Directory to save output files (default: current directory)
    
    Returns:
        reviewer: PredictionReviewer object
    """
    if not os.path.exists(images_dir):
        print(f"Image directory not found: {images_dir}")
        return
    
    if not os.path.exists(predictions_json):
        print(f"Prediction results file not found: {predictions_json}")
        return
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    reviewer = PredictionReviewer(images_dir, predictions_json, filename_prefix, output_dir)
    return reviewer

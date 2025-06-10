from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time

# Set the path to the model directory
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as viz_utils

tf.get_logger().setLevel('ERROR')

PATH_TO_SAVED_MODEL = "./saved_model"
PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Modern TensorFlow GPU memory growth setup (optional, safe for CPU-only too)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs available: ", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Could not set memory growth: {e}")

print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# Store detection history in memory (for demo purposes)
detection_history = []

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']

def inference(image_np):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = detect_fn(input_tensor)

    # Convert outputs to numpy arrays and process
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items() if key != 'num_detections'}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
    # Prepare detection details for UI (top 10)
    detection_details = []
    for i in range(min(10, num_detections)):
        score = detections['detection_scores'][i]
        if score < 0.3:
            continue
        class_id = int(detections['detection_classes'][i])
        label = category_index.get(class_id, {'name': str(class_id)})['name']
        box = detections['detection_boxes'][i].tolist()
        detection_details.append({
            'label': label,
            'score': float(score),
            'box': box
        })
    return image_np_with_detections, detection_details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file', filename=filename))
    # If no file or invalid file, show error or redirect to home
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_np = np.array(Image.open(image_path).convert('RGB'))
    image_np_inferenced, detection_details = inference(image_np)
    im = Image.fromarray(image_np_inferenced)
    im.save(image_path)
    # Save detection details to history (limit to 10)
    detection_history.append({
        'filename': filename,
        'detections': detection_details
    })
    if len(detection_history) > 10:
        detection_history.pop(0)
    # Render result page with detection details and download link
    return render_template('index.html',
        result_image=url_for('uploaded_file', filename=filename, _external=False),
        detection_details=detection_details,
        history=list(reversed(detection_history)),
        show_result=True
    )

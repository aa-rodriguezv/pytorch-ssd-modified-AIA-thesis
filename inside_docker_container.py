import cv2
import datetime
import os
import sys

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor

base_model_dir = 'models/coloredbags'
model_dir = os.path.expanduser(base_model_dir)

# find the checkpoint with the lowest loss
model_path = ''
best_loss = 10000
for file in os.listdir(model_dir):
    if not file.endswith(".pth"):
        continue
    try:
        loss = float(file[file.rfind("-") + 1:len(file) - 4])
        if loss < best_loss:
            best_loss = loss
            model_path = os.path.join(model_dir, file)
    except ValueError:
        continue
print('found best checkpoint with loss {:f} ({:s})'.format(best_loss, model_path))

# Load Detection Model
label_path = base_model_dir + '/labels.txt'
class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

cam_port = 0
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera
result, orig_image = cam.read()

image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)
complete_tuple_box_label_prob_list = []

# Write new Image with Prediction
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
    complete_tuple_box_label_prob_list.append((box[0], box[1], box[2], box[3], labels[i], probs[i]))

now = datetime.now()
dt_format = "%Y_%m_%d_%H_%M_%S"
dt_string = now.strftime(dt_format)
path = f"/jetson-inference/data/coloredbags/detection_results_{dt_string}.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")

# Sort for closest to the band
complete_tuple_box_label_prob_list.sort(key=lambda x: x[0], reverse=True,)

result_label_array = []
for i in range(len(complete_tuple_box_label_prob_list)):
    current_detected_label = complete_tuple_box_label_prob_list[i][4]
    # Base Case
    if i == 0:
        result_label_array.append(current_detected_label)
    # Check if Labels are different (current from previous)
    elif current_detected_label != complete_tuple_box_label_prob_list[i-1][4]:
        result_label_array.append(current_detected_label)
    # Check for Overlapping Boxes with the Same Label
    else:
        current_x1_box_bound = complete_tuple_box_label_prob_list[i][0]
        current_y1_box_bound = complete_tuple_box_label_prob_list[i][1]
        current_x2_box_bound = complete_tuple_box_label_prob_list[i][2]
        current_y2_box_bound = complete_tuple_box_label_prob_list[i][3]

        previous_x1_box_bound = complete_tuple_box_label_prob_list[i-1][0]
        previous_y1_box_bound = complete_tuple_box_label_prob_list[i-1][1]
        previous_x2_box_bound = complete_tuple_box_label_prob_list[i-1][2]
        previous_y2_box_bound = complete_tuple_box_label_prob_list[i-1][3]

        current_xmin_box_bound = min(current_x1_box_bound, current_x2_box_bound)
        current_ymin_box_bound = min(current_y1_box_bound, current_y2_box_bound)
        current_xmax_box_bound = max(current_x1_box_bound, current_x2_box_bound)
        current_ymax_box_bound = max(current_y1_box_bound, current_y2_box_bound)

        previous_xmin_box_bound = min(previous_x1_box_bound, previous_x2_box_bound)
        previous_ymin_box_bound = min(previous_y1_box_bound, previous_y2_box_bound)
        previous_xmax_box_bound = max(previous_x1_box_bound, previous_x2_box_bound)
        previous_ymax_box_bound = max(previous_y1_box_bound, previous_y2_box_bound)


        def is_overlapping_1d(box1, box2):
            return box1[1] >= box2[0] and box2[1] >= box1[0]


        current_box1 = {
            'x': (current_xmin_box_bound, current_xmax_box_bound),
            'y': (current_ymin_box_bound, current_ymax_box_bound)
        }
        previous_box2 = {
            'x': (previous_xmin_box_bound, previous_xmax_box_bound),
            'y': (previous_ymin_box_bound, previous_ymax_box_bound)
        }
        overlap_x_axis = is_overlapping_1d(current_box1.x, previous_box2.x)
        overlap_y_axis = is_overlapping_1d(current_box1.y, previous_box2.y)
        boxes_overlap = overlap_x_axis and overlap_y_axis

        if boxes_overlap:
            current_box1_x_size = current_xmax_box_bound - current_xmin_box_bound
            current_box1_y_size = current_ymax_box_bound - current_ymin_box_bound
            current_box1_area = current_box1_x_size * current_box1_y_size

            previous_box2_x_size = previous_xmax_box_bound - previous_xmin_box_bound
            previous_box2_y_size = previous_ymax_box_bound - previous_ymin_box_bound
            previous_box2_area = previous_box2_x_size * previous_box2_y_size

            max_box_area = max(current_box1_area, previous_box2_area)

            max_of_min_x_bound = max(current_xmin_box_bound, previous_xmin_box_bound)
            max_of_min_y_bound = max(current_ymin_box_bound, previous_ymin_box_bound)
            min_of_max_x_bound = min(current_xmax_box_bound, previous_xmax_box_bound)
            min_of_max_y_bound = min(current_ymax_box_bound, previous_ymax_box_bound)

            overlapping_box_x_size = min_of_max_x_bound - max_of_min_x_bound
            overlapping_box_y_size = min_of_max_y_bound - max_of_min_y_bound
            overlapping_box_area = overlapping_box_x_size * overlapping_box_y_size

            how_much_overlap = overlapping_box_area / max_box_area
            # Area of Overlapping Box represents less than 50% of the Maximum Box Area (is not representative)
            if how_much_overlap < 0.5:
                result_label_array.append(current_detected_label)
        # They do not overlap so they can be safely added
        else:
            result_label_array.append(current_detected_label)

result_path = '/jetson-inference/data/' + 'bags_trial.txt'
if os.path.exists(result_path):
    os.remove(result_path)

with open(result_path, 'w') as filehandle:
    filehandle.write('\n'.join(result_label_array))
print('Job Success')



import carla
import numpy as np
import pygame
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import os
import csv
import time
import collections
from torch import nn
import timm

# ==============================================================================
# -- Configuration & Model Selection -------------------------------------------
# ==============================================================================

# Change this to 'fasterrcnn' or 'vit' to switch models
MODEL_TO_USE = "fasterrcnn"  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.4
FPS_LOG_INTERVAL = 10  # Log performance every 10 frames

# ==============================================================================
# -- Model Definitions ---------------------------------------------------------
# ==============================================================================

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def load_mobilevit_model():
    """Loads a conceptual MobileViT object detection model."""
    print("✅ Loaded MobileViT (Simulated with a real backbone).")
    try:
        model_backbone = timm.create_model('mobilevit_s', pretrained=True)
    except Exception as e:
        print(f"Error loading timm model: {e}. Please ensure `timm` is installed.")
        return None, None

    class MobileViTDetection(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(1000, 100)
            
        def forward(self, img_tensor):
            features = self.backbone(img_tensor.unsqueeze(0))
            _ = self.head(features)
            boxes = torch.rand(2, 4).to(img_tensor.device) * 800
            labels = torch.randint(1, len(COCO_INSTANCE_CATEGORY_NAMES), (2,)).to(img_tensor.device)
            scores = torch.rand(2).to(img_tensor.device) * 0.5 + 0.5
            return [{'boxes': boxes, 'labels': labels, 'scores': scores}]
    
    model = MobileViTDetection(model_backbone)
    model.to(DEVICE)
    model.eval()
    return model, COCO_INSTANCE_CATEGORY_NAMES

def load_fasterrcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1').to(DEVICE)
    model.eval()
    print("✅ Loaded Faster R-CNN.")
    return model, COCO_INSTANCE_CATEGORY_NAMES

def run_model(rgb_img, model, model_type):
    start_time = time.time()
    detections = []
    
    img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    img_tensor = F.to_tensor(img).to(DEVICE)
    
    with torch.no_grad():
        if model_type == 'vit':
            output = model(img_tensor)[0]
        else:
            output = model([img_tensor])[0]

    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            detections.append([class_name, float(score), x1, y1, x2, y2])

    inference_time = (time.time() - start_time) * 1000
    
    det_img = rgb_img.copy()
    for det in detections:
        class_name, score, x1, y1, x2, y2 = det
        cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_img, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return det_img, detections, inference_time

# ==============================================================================
# -- CARLA Setup ---------------------------------------------------------------
# ==============================================================================

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

for actor in world.get_actors().filter('*'):
    if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
        actor.destroy()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print("✅ Vehicle spawned successfully!")

# RGB Camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Semantic Segmentation Camera
sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
sem_bp.set_attribute('image_size_x', '800')
sem_bp.set_attribute('image_size_y', '600')
sem_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
sem_camera = world.spawn_actor(sem_bp, sem_transform, attach_to=vehicle)

# LiDAR
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# --- Sensor Data Buffers ---
latest_frame = None
latest_sem_frame = None
latest_lidar = None
frame_count = 0
log_path = os.path.join(os.getcwd(), "performance_log.csv")
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "inference_ms", "fps", "frame_count"])

def process_image(image):
    global latest_frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    latest_frame = array[:, :, :3]
camera.listen(process_image)

def process_sem_image(image):
    global latest_sem_frame
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    latest_sem_frame = array[:, :, :3]
sem_camera.listen(process_sem_image)

def process_lidar(point_cloud):
    global latest_lidar
    latest_lidar = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4))
lidar.listen(process_lidar)

# --- Manual Control Setup ---
pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("CARLA Manual Data Capture")
clock = pygame.time.Clock()

print("""
Controls:
  W = Throttle | S = Brake/Reverse | A = Left | D = Right | Q = Quit
""")
print(f"Loading '{MODEL_TO_USE}' model for real-time visualization...")

# ==============================================================================
# -- Main Simulation Loop ------------------------------------------------------
# ==============================================================================
try:
    if MODEL_TO_USE == "fasterrcnn":
        model, class_names = load_fasterrcnn_model()
        model_name = "Faster R-CNN"
    elif MODEL_TO_USE == "vit":
        model, class_names = load_mobilevit_model()
        model_name = "MobileViT"

    while latest_frame is None or latest_sem_frame is None or latest_lidar is None:
        print("Waiting for sensor data...")
        time.sleep(0.1)
    
    running = True
    while running:
        world.tick()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("\nExiting simulation.")
                    running = False
        
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()
        if keys[pygame.K_w] or keys[pygame.K_UP]: control.throttle = 1.0; control.brake = 0.0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: control.throttle = 0.0; control.brake = 1.0
        else: control.throttle = 0.0; control.brake = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: control.steer = -0.5
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: control.steer = 0.5
        else: control.steer = 0.0
        vehicle.apply_control(control)

        det_img, _, inference_time = run_model(latest_frame.copy(), model, MODEL_TO_USE)
        cv2.putText(det_img, f"Throttle: {control.throttle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(det_img, f"Steer: {control.steer:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(det_img, f"Brake: {control.brake:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Driver View (Real-time detection)", det_img)
        
        frame_count += 1
        if frame_count % FPS_LOG_INTERVAL == 0:
            fps = 1000 / inference_time if inference_time > 0 else float('inf')
            print(f"Model: {model_name:<15} | Frame: {frame_count:<5} | Inference: {inference_time:.2f} ms | FPS: {fps:.2f}", end='\r')
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, inference_time, fps, frame_count])

        if latest_sem_frame is not None:
            cv2.imshow("Semantic Segmentation", latest_sem_frame)
        if latest_lidar is not None:
            lidar_vis = np.zeros((400, 400, 3), dtype=np.uint8)
            if latest_lidar.shape[0] > 0:
                points = latest_lidar[:, :2]
                points = np.clip((points * 4) + 200, 0, 399).astype(np.int32)
                lidar_vis[points[:,1], points[:,0]] = (0,255,0)
            cv2.imshow("LiDAR", lidar_vis)
        
        if cv2.waitKey(1) == 27:
            break

finally:
    if 'vehicle' in locals() and vehicle.is_alive: vehicle.destroy()
    if 'camera' in locals() and camera.is_alive: camera.destroy()
    if 'sem_camera' in locals() and sem_camera.is_alive: sem_camera.destroy()
    if 'lidar' in locals() and lidar.is_alive: lidar.destroy()
    pygame.quit()
    cv2.destroyAllWindows()
    print("\nSimulation ended and resources cleaned up.")
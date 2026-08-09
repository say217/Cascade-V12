## Flood Detection Module

| **Flood Detection** |
|:---:|
| <img width="866" height="410" alt="Screenshot 2026-08-08 115459" src="https://github.com/user-attachments/assets/6f64a8a8-5329-4062-8fcc-6dd0a51e1cc3" /> | 


The drone-based flood detection engine automates the critical process of monitoring large-scale disaster zones by utilizing a fine-tuned YOLOv12 object detection model to instantly scan video streams, bypassing the limitations of human fatigue and restricted operational scale. In active emergency scenarios, manually tracking thousands of hours of aerial footage is structurally impossible, leading to critical delay blind spots in mapping submerged infrastructure. This system solves that exact bottlenecks by transforming raw pixel data into real-time, quantifiable bounding coordinates, enabling emergency dispatch teams to instantly identify flooded roadways, track rising water boundaries dynamically, and deploy life-saving rescue assets with automated, machine-level speed and precision.

## Traing Configurations

The choice of YOLOv8 Medium Segmentation (yolov8m-seg.pt) provides an optimal balance between high feature-extraction capacity and real-time operational efficiency for drone-based aerial monitoring. Choosing a unified instance segmentation architecture allows the system to perform simultaneous object detection (bounding boxes) and pixel-level segmentation (polygon masks) in a single forward pass, eliminating the computational overhead of running separate detection and mask generation pipelines. The medium (m) variant provides sufficient layer depth and parameters to resolve complex, low-contrast water boundaries and small flood pockets without incurring the severe latency or memory footprint associated with larger models like yolov8x-seg.Training ConfigurationResolution and Batching (imgsz=640, batch=16): An image size of $640\times640$ maintains spatial clarity for small aerial targets while keeping GPU memory usage well within the limits of a single Tesla T4 GPU. A batch size of 16 ensures stable gradient updates during backpropagation without risking out-of-memory errors on Kaggle.Optimization & Convergence (lr0=0.01, lrf=0.01, warmup_epochs=3.0, patience=6): The training pipeline uses an initial learning rate ($\text{lr}_0$) of $0.01$ decaying to a final fraction ($\text{lrf}$) of $0.01$, guided by a $3.0$-epoch warm-up schedule to stabilize initial gradient updates. Early stopping with patience=6 halts training automatically if validation performance plateaus for 6 consecutive epochs, preventing overfitting and saving compute time.Spatial & Aerial Augmentations (degrees=15.0, flipud=0.5, mosaic=1.0, mixup=0.15): To adapt to top-down drone perspectives, vertical flips (flipud=0.5) and rotational variance (degrees=15.0) are applied to enforce orientation invariance. Mosaic augmentation (mosaic=1.0) forces the network to locate smaller, distant flood pockets within composite frames, while MixUp (mixup=0.15) blends image textures to improve boundary differentiation along murky or soft water edges.Loss Balance & Hardware Control (box=7.5, cls=0.5, dfl=1.5, amp=True, cache=False): Bounding box loss gain is emphasized (box=7.5, dfl=1.5) over classification loss (cls=0.5) since the dataset targets a single class (flood) where precise spatial localization is paramount. Automatic Mixed Precision (amp=True) accelerates matrix operations using FP16 precision on CUDA cores, while RAM caching is disabled (cache=False) to guarantee stability across multi-dataset loads.


### Datasetswe use in training

```bash
https://universe.roboflow.com/long-nguyen-hoang-9ecmq/flood-4oe1x

https://universe.roboflow.com/ta-r2l28/bencana

https://universe.roboflow.com/reaserch/flood-area-segmentation-biizb

https://universe.roboflow.com/flood-uwifo/flood-1sljl

```


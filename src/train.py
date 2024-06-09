from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data = "data/animals.yaml"
            , epochs = 150
            , batch = -1
            , optimizer = "AdamW"
            , lr0 = 0.00717
            , lrf = 0.00828
            , momentum = 0.88802
            , weight_decay = 0.0005
            , warmup_epochs = 2.79803
            , warmup_momentum = 0.895
            , box = 6.87221
            , cls = 0.50216
            , dfl = 1.51706
            , hsv_h = 0.01507
            , hsv_s = 0.55786
            , hsv_v = 0.47574
            , degrees = 0.0
            , translate = 0.10493
            , scale = 0.52712
            , shear = 0.0
            , perspective = 0.0
            , flipud = 0.0
            , fliplr = 0.49317
            , bgr = 0.0
            , mosaic = 1.0
            , mixup = 0.0
            , copy_paste = 0.0 )  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

## GROUP 412, FRIDAY 10am ##

Files attached:
    - TargetPoseEst.py
    - yolov5/
    - model.best.pt

File directory Format
Week06-07/
    - yolov5/
    - lab_output/img_0....
    - TargetPoseEst.py
    - CV_eval.py
    - network/scripts/model/model.best.pt

Instructions:
    1. Place all the files in the correct locations specified above
    2. cd yolov5/
    3. pip install -r Requirements.txt ## torch.__version__ > 1.7 is required for YOLO
    4. cd ..
    5. python TargetPoseEst.py ## python3 required
    6. python CV_eval.py ## python3 required

Absolute paths in our files, in case of any errors on import:
    TargetPoseEst_YOLO: __main__: Line 218
        with open(base_dir / 'lab_output/images.txt') as fp:
    TargetPoseEst_YOLO: get_image_info(): Line 44
        ckpt = os.getcwd() + '/network/scripts/model/model.best.pt'
    TargetPoseEst_YOLO: get_image_info(): Line 47
        model = torch.hub.load('./yolov5', 'custom', path=ckpt, source='local')

Contact information:
    Aidan Pritchard, apri0009@student.monash.edu, 0468572518
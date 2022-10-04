## GROUP 412, FRIDAY 10am ##

Files attached:
    - calibration/
    - TargetPoseEst.py
    - yolov5/
    - model.best.pt

File directory Format
Week06-07/
    - calibration/ # add this
    - yolov5/ # add this
    - lab_output/img_0.... # for reference, may need to be moved from inside Example_Dataset equivalent
    - TargetPoseEst.py # add this
    - CV_eval.py # for reference
    - network/scripts/model/model.best.pt # add this, may need to add model folder

Instructions:
    1. Place all the files in the correct locations specified above
    2. cd yolov5/
    3. pip install -r requirements.txt ## torch.__version__ > 1.7 is required for YOLO
    4. pip install scikit-learn
    5. cd ..
    6. python3 TargetPoseEst.py 
    7. python3 CV_eval.py 

Absolute paths in our files, in case of any errors on import:
    TargetPoseEst_YOLO: __main__: Line 218
        with open(base_dir / 'lab_output/images.txt') as fp:
    TargetPoseEst_YOLO: get_image_info(): Line 44
        ckpt = os.getcwd() + '/network/scripts/model/model.best.pt'
    TargetPoseEst_YOLO: get_image_info(): Line 47
        model = torch.hub.load('./yolov5', 'custom', path=ckpt, source='local')

Contact information:
    Aidan Pritchard, apri0009@student.monash.edu, 0468572518
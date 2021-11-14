



# Object Detection in Aerial Image

# Result

 1.   `hw2_R07921052.pdf` with outcomes and experiments. 

<p align="center">
  <img width="550" height="500" src="">
</p>

For more details, please view `hw2_R07921052.pdf` for results and experiments.

### Evaluation
To evaluate your model, you can run the provided evaluation script provided in the starter code by using the following command.

    python3 hw2_evaluation_task.py <PredictionDir> <AnnotationDir>

 - `<PredictionDir>` should be the directory to output your prediction files (e.g. `hw2_train_val/val1500/labelTxt_hbb_pred/`)
 - `<AnnotationDir>` should be the directory of ground truth (e.g. `hw2_train_val/val1500/labelTxt_hbb/`)

Note that your predicted label file should have the same filename as that of its corresponding ground truth label file (both of extension ``.txt``).

### Visualization
To visualization the ground truth or predicted bounding boxes in an image, you can run the provided visualization script provided in the starter code by using the following command.

    python3 visualize_bbox.py <image.jpg> <label.txt>

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw2_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading Policy*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw2.sh`  
The shell script file for running your `YoloV1-vgg16bn` model.
 3.   `hw2_best.sh`  
The shell script file for running your improved model.

We will run your code in the following manner:

    bash ./hw2.sh $1 $2
    bash ./hw2_best.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images`), and `$2` is the output prediction directory (e.g. `test/labelTxt_hbb_pred/` ).


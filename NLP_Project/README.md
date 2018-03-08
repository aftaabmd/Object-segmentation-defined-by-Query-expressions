Software Pre-requisite:
1. Tensorflow (v7.0 or higher)
2. Anaconda (to include these packages numpy, skimage, json, Queue, threading, os, pyximport, scipy, re)

Demo:
1. For the demo purpose, trained models can be downloaded from this location

https://umass0-my.sharepoint.com/personal/ravichoudhar_umass_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=O8PDYPbiZ8N8UMPBySXnazr6MXLoLslxEYPHMiV7PyU%3d&folderid=2_1ab930c0772c04965825a449ae46e3e9f&rev=1

and place this tfmodel folder inside Project/exp-referit folder.

2. Run this python demo file as: "python /demo/text_objseg_demo.py" in terminal with Project as base.


In order to run from scratch for training and evaluation on referit dataset(NOTE: It would take 4-5 days for the model to be trained)
1. Download dataset and VGG network: We can download dataset and vgg architecture by running following scripts in terminal "exp-referit/referit-dataset/download_referit_dataset.sh" and "models/convert_caffemodel/params/download_vgg_params.sh". Once dataset script is over, extract referit_edgeboxes_top100.zip which is inside "exp-referit/data" folder

2. In order to make this package available to run, add the repository root directory to Python's module path: "export PYTHONPATH=.:$PYTHONPATH" 

3. Get batches of bounding boxes by running : python exp-referit/build_training_batches_det.py

4. Get segmentation batches using: python exp-referit/build_training_batches_seg.py

5. Select the GPU during training: export GPU_ID=<gpu id>. Use 0 for <gpu id> since we had one GPU on our machine.

6. Train the language-based bounding box localization model: python exp-referit/exp_train_referit_det.py $GPU_ID

7. Train the low resolution language-based segmentation model (use weights and bias variable from the previous bounding box localization model): python exp-referit/init_referit_seg_lowres_from_det.py && python exp-referit/exp_train_referit_seg_lowres.py $GPU_ID

8. Train the high resolution language-based segmentation model (use weights and bias variable from the previous low resolution segmentation model): python exp-referit/init_referit_seg_highres_from_lowres.py && python exp-referit/exp_train_referit_seg_highres.py $GPU_ID


Evaluation

9. Run evaluation for the high resolution language-based segmentation model:
python exp-referit/exp_test_referit_seg.py $GPU_ID
This should reproduce the results in the paper.

10. We can also evaluate the language-based bounding box localization model:
python exp-referit/exp_test_referit_det.py $GPU_ID

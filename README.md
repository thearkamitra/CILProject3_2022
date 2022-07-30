# CILProject3_2022

Third project topic for CIL 2022 - Road Segmentation

  

This is a python program that trains a model for road segementation and can use that model to predict roads in new aerial images as well. Different models and losses have been used which can be configured when running the program, look at the relavanat section for the different params usable when running this.

  
  

## Training the model

To train the model, the main has to be run with the "--cmd train" param, for example:

`python main.py --cmd train`


This will run the training part of the model with the given input data which needs to be stored at 'training/images' and 'training/groundtruth'. The model will be saved in the same folder as a pth file with the vhosen model name along with the timestamp and that can be used later to generate predictions.

  

### Additional prameters for training

Training can be additionally configured using the following commands:
- \-e | --epochs	: An integer to specify the number of epochs to run
- \-b | --batch		: An integer specifying the batch size
- \--lr 					: Specify the initial learning rate. Default is 1e-4
- \- -model : Choose which model architecture to use. Can be one of the following: baseline, fcn_res, unet, deeplabv3 and segformer
- \-l | --loss : Choice of loss function to be used by the model. Can be one of the following: dice (Dice Loss), wbce(Static Weighted BCE), wbce2 (Dynamic Weighted BCE), bbce (Border Weighted BCE), focal (Focal Loss), tv (Tversky Loss)
- \-p | --modeltoload : Name of saved model pth if you want to reuse a previously trained model
- \- -pretrain: Boolean value specifying if model should first be pretrain on a different dataset.
- \--wandb : Boolean value specifying if runs should be logged to wandb or not
- \-w | --warmup_steps: Since the learning rate dynamically adjusts to the loss value, this helps fix the initial lr for a specified number of epochs. Useful for pretraining
- \-sp | --save_path : Location where model should be saved

Post training the model is saved as a pth file in the given save path location.


## Making Predictions
To make predictions, you should have the pth file from a model previously trained using this program.  With that you can use the 'test' command parameter to run the model for predictions, like:
`python main.py --cmd test -p fcn_res-dice-06-27_17-58.pth`

The input images for the predictions need to be stored in 'test/images' folder. All predictions are saved in the 'test/predictions' folder.

### Additional params for prediction
The predictions can be additionally configured using the following properties:
- \-p | --modeltoload: Name of saved model pth to reuse a previously trained model
- -model : Need to specify the same model as the one saved in the .pth file. Can be one of the following: baseline, fcn_res, unet, deeplabv3 and segformer

You can view the predicted images in the 'test/predictions' folder
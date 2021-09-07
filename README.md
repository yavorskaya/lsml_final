# Final project LSML:

### Metal Surface Defects Classification
This project is written to classify metal surface defects in XRay images  
It has web gui.

Images: 
web: flask web app

### Data description
The dataset was downloaded from Kaggle (https://www.kaggle.com/fantacher/neu-metal-surface-defects-data). It contains six kinds of typical surface defects of the hot-rolled steel strip:
1. Crazing
2. Inclusion
3. Patches
4. Pitted surface
5. Rolled-in scale
6. Scratches. 
The database consists of 1,800 grayscale images, which are devided into train, valid and test datasets.

### Model description

model: CNN,
optimizer: AdamW, 
loss: CrossEntropyLoss, 
metric: Accuracy

### User instructions

installation
https://github.com/yavorskaya/lsml_final.git

running app
docker-compose up --build

then open
http://localhost:5000 (http://localhost:5000/home)

upload a file
click 'upload'

the results will be presented at http://localhost:5000/prediction

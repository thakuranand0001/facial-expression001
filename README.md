# Facial Expression Classification (FER)

A 7-class facial-expression image-classification project based on the uploaded `4-img-classification.ipynb` and Streamlit `app.py` references. The original `app.py` was for chest X-rays, so this version has been adapted to the supplied facial-expression dataset.

## Classes

`angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

The dataset contains alternate folder spellings (`disgust`/`disgusted`, `fear`/`fearful`, and `surprise`/`surprised`). `train_model.py` maps each pair to one canonical class.

## Dataset layout

1. Extract `FER-56-images-per-class.zip`.
2. The expected path is:

```text
FER-56-images-per-class/
  facial_expression_source/
    train/
      angry/ ...
      disgust/ ...
      disgusted/ ...
      fear/ ...
      fearful/ ...
      happy/ ...
      neutral/ ...
      sad/ ...
      surprise/ ...
      surprised/ ...
    test/
      angry/ ...
      disgusted/ ...
      fearful/ ...
      happy/ ...
      neutral/ ...
      sad/ ...
      surprised/ ...
```

The script discovers images from the folders and combines aliases. It creates a stratified 80/20 training/validation split from `train/`; the supplied `test/` split is held out for final evaluation.

## Setup (Windows PowerShell)

```powershell
cd fer_expression_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

Run from the project directory. Update the dataset path if you extracted it elsewhere:

```powershell
python train_model.py --data "../FER-56-images-per-class/FER-56-images-per-class/facial_expression_source" --epochs 30
```

If the extracted folder is named differently, pass the path that directly contains `train` and `test`.

Training creates:
- `facial_expression_model.keras`
- `classes.json`
- `training_history.json`

## Run the Streamlit app

After training:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal and upload a JPG, JPEG, or PNG image.

## Notes

- The network uses 64×64 RGB inputs, convolution/pooling blocks, dropout, and a seven-unit softmax output.
- Class weighting helps address uneven class counts; augmentation is applied only to training batches.
- Facial-expression labels are not reliable measurements of a person's internal emotional state. This project is for educational experimentation only.

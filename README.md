## captcha_solver
This project is to solve captcha image using Tensorflow for Machine learning.

## Model flow
1. Generate captcha images
2. preprocessing
3. Learning
4. Evaluation
5. save model


## Structures
- generate_captcha.py
- preprocess.py
- cnn.py
- main.py
- data/
- font/

### generate_captcha.py
Generate the captcha data set.
I used 5 length cpathca data using the font in the directory `./font`

### preprocess.py
Parse the catpcha image data from the directory `./data` and return the result as numpy array.
X_data: (N, 60, 160, 3)
Y_data: (N, 5, 10)


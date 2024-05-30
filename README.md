# How to use (Ubuntu 22.04)
Run EdgeSignalDemo.bin (slow startup is expected as pyinstaller starts)
"q" to shutdown early

# How to use (python 3.10)
Activate venv
pip install -r requirements.txt
python main.py

# Notes:
- Person is considered to have crossed the midline when the top of their bbox crosses
  because the head tends to be stable
- Right side of image was used to training the gender model
- Training for the gender detection is poor for a couple of reasons:
  - Small dataset
  - Blurry images made blurrier by the resize
  - Difficulty for a human (me) to sort training data -> it's somewhat of a long hair detector
  - Low amount of iterations (model did converge pretty quickly though)
- No specific reason for black and white video footage. It was the best video with pretty high
  foot traffic, good angle and visually distinct genders. Color footage works too.

# Credits
mall.mp4: https://pixabay.com/videos/shopping-center-escalators-modern-12876/
mall-2.mp4: https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/
mall-3.mp4: https://www.pexels.com/video/a-day-at-the-mall-1338598/

yolo-v9 - implementation of paper:
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}

norfair:
@software{joaquin_alori_2023_7504727,
  author       = {Joaquín Alori and
                  Alan Descoins and
                  javier and
                  Facundo Lezama and
                  KotaYuhara and
                  Diego Fernández and
                  Agustín Castro and
                  fatih and
                  David and
                  Rocío Cruz Linares and
                  Francisco Kurucz and
                  Braulio Ríos and
                  shafu.eth and
                  Kadir Nar and
                  David Huh and
                  Moises},
  title        = {tryolabs/norfair: v2.2.0},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v2.2.0},
  doi          = {10.5281/zenodo.7504727},
  url          = {https://doi.org/10.5281/zenodo.7504727}
}

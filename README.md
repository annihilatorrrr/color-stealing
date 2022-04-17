<h1 align="center">
  <b>Color Stealing</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" /></a>
      <a href= "LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-white.svg" /></a>
</p>

Python implementation of the [Color Stealing algorithm](https://maths-people.anu.edu.au/~barnsley/pdfs/fractal_tops.pdf) for fractals. In short, this technique constructs aesthetically pleasing fractals by copying color patterns from a given auxiliary image.



<p align="center">
  <image src="assets/generated/lava_22_1.png" />
  Definitely not a cherry-picked sample.
</p>






|Auxiliary Image | Binary Fractal | Color Stealing |
| :--: |:--:|:--:|
|<img src="assets/auxiliary/lava.jpg" width="500" height="300"/>|<img src="templates/images/27.png" width="500" height="300" />|<img src="assets/generated/lava_27_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/lava.jpg" width="500" height="300"/>|<img src="templates/images/5.png" width="500" height="300" />|<img src="assets/generated/lava_5_2.png" width="500" height="300"/>|
|<img src="assets/auxiliary/lava.jpg" width="500" height="300"/>|<img src="templates/images/34.png" width="500" height="300" />|<img src="assets/generated/lava_34_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/lsd2.jpg" width="500" height="300"/>|<img src="templates/images/43.png" width="500" height="300" />|<img src="assets/generated/lsd2_43_6.png" width="500" height="300"/>|
|<img src="assets/auxiliary/lsd2.jpg" width="500" height="300"/>|<img src="templates/images/30.png" width="500" height="300" />|<img src="assets/generated/lsd2_30_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/color.jpg" width="500" height="300"/>|<img src="templates/images/22.png" width="500" height="300" />|<img src="assets/generated/color_22_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/color.jpg" width="500" height="300"/>|<img src="templates/images/27.png" width="500" height="300" />|<img src="assets/generated/color_27_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/van_gogh.jpg" width="500" height="300"/>|<img src="templates/images/39.png" width="500" height="300" />|<img src="assets/generated/van_gogh_39_1.png" width="500" height="300"/>|
|<img src="assets/auxiliary/texture1.jpg" width="500" height="300"/>|<img src="templates/images/42.png" width="500" height="300" />|<img src="assets/generated/texture1_42_2.png" width="500" height="300"/>|
|<img src="assets/auxiliary/wheat.jpg" width="500" height="300"/>|<img src="templates/images/43.png" width="500" height="300" />|<img src="assets/generated/wheat_43_2.png" width="500" height="300"/>|











Table of contents
===

<!--ts-->
  * [➤ Paper Summary](#paper-summary)
    * [➤ GAN](#gan)
    * [➤ GUI](#gui)
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
  * [➤ Citations](#citations)
<!--te-->


<a  id="paper-summary"></a>
Paper Summary
===

This paper combines generative adversarial networks with interactive evolutionary computation. Specifically,
instead of randomly sampling from gans, a user can guide the generation by selecting images with desired traits using an interactive gui.

<a  id="gan"></a>
GAN
---
The author of this repo does not possess the hardware, the time, the patience or the skills necessary to train gans. Threfore, the pretrained models from [Facebook's GAN zoo](https://github.com/facebookresearch/pytorch_GAN_zoo) are employed.



<a  id="gui"></a>
GUI
---
Whereas the authors of the paper developed a web interface to display images, the author of this repo possesses zero web development skills and therefore makes due with a makeshift [tkinter](https://docs.python.org/3/library/tkinter.html) gui.



<a  id="installation"></a>
Installation
===
```
$ git clone https://github.com/davidsvy/color-stealing
$ cd color-stealing
$ pip install -r requirements.txt
```



<a  id="usage"></a>
Usage
===

```
$ python run.py [-c configs/config.yaml]
```

<a  id="citations"></a>
Citations
===

```bibtex
@misc{bontrager2018deep,
      title={Deep Interactive Evolution}, 
      author={Philip Bontrager and Wending Lin and Julian Togelius and Sebastian Risi},
      year={2018},
      eprint={1801.08230},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}

```


<h1 align="center">
  <b>Color Stealing</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" /></a>
      <a href= "LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-white.svg" /></a>
</p>

Python implementation of the [Color Stealing algorithm](https://maths-people.anu.edu.au/~barnsley/pdfs/fractal_tops.pdf). In short, this technique constructs aesthetically pleasing fractals by copying color patterns from a given auxiliary image to a binary fractal.



<p align="center">
  <image src="assets/generated/lava_22_1.png" />
  Definitely not a cherry-picked sample.
</p>




Table of contents
===

<!--ts-->
  * [➤ Summary](#summary)
  * [➤ Generated Samples](#generated-samples)
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
  * [➤ Contact](#contact)
  * [➤ Citations](#citations)
<!--te-->


<a  id="summary"></a>
Summary
===

Perhaps the most simple method to construct a fractal image is the [Chaos Game](https://en.wikipedia.org/wiki/Chaos_game). A fractal can be defined by its [iterated function system](https://en.wikipedia.org/wiki/Iterated_function_system) (IFS), which is a set of n functions. For images, each function is $F_i: R^2 → R^2$ with a corresponding probability $p_i$. The Chaos Game begins by sampling a random 2d coordinate $x_0 \in R^2$. At the j-th iteration, an integer $k_j$ is sampled based on the distribution $p$. The next coordinate is computed as $x_j = F_{k_j}(x_{j-1})$. This coordinate is then plotted on a 2d grid, which was initially empty. After a number of iterations, the result is a binary image. Examples of such images can be found in the second column of the [generated samples](#generated-samples) as well as at `templates/images`.

The [Color Stealing algorithm](https://maths-people.anu.edu.au/~barnsley/pdfs/fractal_tops.pdf) extends the Chaos Game by adding color to the constructed fractal. This requires an auxiliary image as input. Specifically, two IFSs are kept: one with functions $F_i$ and probabilities $p_i$ for the fractal image and one with functions $G_i$ and no probabilities for the auxiliary image. Similarly, two coordinates are initially sampled: $x_0, z_0 \in R^2$. At the j-th iteration, an integer $k_j$ is likewise sampled from the distribution $p$, which is followed by calculations $x_j = F_{k_j}(x_{j-1})$ and $z_j = G_{k_j}(z_{j-1})$. Then the fractal is extended as fractal[$x_j$] = image[$z_j$]. This process eventually results in images such as the third column of [generated samples](#generated-samples).




<a  id="generated-samples"></a>
Generated samples
===
Below is a list of examples constructed by the provided scripts. Each sample was generated using the command `python run.py -s -i 100000 -nt -l <ifs_path> -c <img_path>`, where `<ifs_path>` was chosen from `templates/ifs` and `<img_path>` from `assets/auxiliary`. Running this commands produces a different result each time, due to rng. Obviously, all the examples are cherry-picked.

|Auxiliary Image | Binary Fractal | Color Stealing |
| :--: |:--:|:--:|
|<img src="assets/auxiliary/lava.jpg" width="500" height="240"/>|<img src="templates/images/27.png" width="500" height="240" />|<img src="assets/generated/lava_27_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/lava.jpg" width="500" height="240"/>|<img src="templates/images/5.png" width="500" height="240" />|<img src="assets/generated/lava_5_2.png" width="500" height="240"/>|
|<img src="assets/auxiliary/lava.jpg" width="500" height="240"/>|<img src="templates/images/34.png" width="500" height="240" />|<img src="assets/generated/lava_34_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/lsd2.jpg" width="500" height="240"/>|<img src="templates/images/43.png" width="500" height="240" />|<img src="assets/generated/lsd2_43_6.png" width="500" height="240"/>|
|<img src="assets/auxiliary/lsd2.jpg" width="500" height="240"/>|<img src="templates/images/30.png" width="500" height="240" />|<img src="assets/generated/lsd2_30_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/color.jpg" width="500" height="240"/>|<img src="templates/images/22.png" width="500" height="240" />|<img src="assets/generated/color_22_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/color.jpg" width="500" height="240"/>|<img src="templates/images/27.png" width="500" height="240" />|<img src="assets/generated/color_27_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/van_gogh.jpg" width="500" height="240"/>|<img src="templates/images/39.png" width="500" height="240" />|<img src="assets/generated/van_gogh_39_1.png" width="500" height="240"/>|
|<img src="assets/auxiliary/texture1.jpg" width="500" height="240"/>|<img src="templates/images/42.png" width="500" height="240" />|<img src="assets/generated/texture1_42_2.png" width="500" height="240"/>|
|<img src="assets/auxiliary/wheat.jpg" width="500" height="240"/>|<img src="templates/images/43.png" width="500" height="240" />|<img src="assets/generated/wheat_43_2.png" width="500" height="240"/>|



<a  id="installation"></a>
Installation
===
```
git clone https://github.com/davidsvy/color-stealing
cd color-stealing
pip install -r requirements.txt
```



<a  id="usage"></a>
Usage
===
```
python run.py OPTIONS
```

## Saving and rendering:
    -s, --save [FILE_NAME]           If provided, saves the image and the IFS.
                                     If FILE_NAME is not given, the image will 
                                     be saved at DIR/images/TIMESTEP.png and
                                     the IFS at DIR/ifs/TIMESTEP.csv. DIR is
                                     the next argument and TIMESTEP is the
                                     timestep at the time of saving the files. If
                                     FILE_NAME is given, the image will be saved at 
                                     DIR/images/FILE_NAME.png an the IFS at 
                                     DIR/ifs/FILE_NAME.csv. For the script to run, 
                                     either -p or -s must be provided.

    -d, --dir DIR                    Directory where the image & the IFS file will
                                     be saved.

    -p, --plot                       If given, the constructed fractal will be
                                     rendered using plt.imshow(). For the script
                                     to run, either -p or -s must be provided.

    -cm, --cmap CMAP                 cmap for for plt.imshow(). Accepted values are
                                     'i', 'p', 'm', 'c', 'v'. Default value is 'i'.
                                     CMAP is ignored if either -f or -c is provided.


## Fractal parameters:
    -r, --resolution RES             Resolution of the square fractal image. Default
                                     value is 1024.

    -b, --batch_size BS              Number of points calculated at each iteration.
                                     If -f is given, setting BS <= 4 might result in 
                                     better visuals. Default value is 128.

    -i, --n_iter N_ITER              Number of iterations for the chaos game algorithm.
                                     Default value is 50000.

    -ni, --n_ignore N_IGNORE         Number of initial iterations that will not be 
                                     rendered. Default value is 200.

    -nt, --no_tqdm                   If given, the tqdm progress bar will not be used.


## Loading saved IFS:
    -l, --load PATH                  If not provided, the scripts will sample a random IFS.
                                     If provided, the fractal stored inside PATH will be 
                                     generated. PATH must be a path to a valid IFS csv 
                                     file.

    -m, --mutate PATH                Same as the previous argument, with the difference
                                     that the loaded IFS will be slightly mutated in a 
                                     random fashion.

      
## Color Stealing:
    -c, --color_steal PATH           If not provided (and -f not provided), a binary 
                                     fractal will be generated. If provided, the fractal 
                                     will be constructed by copying color patterns from 
                                     the image stored at PATH using the Color Stealing
                                     algorithm.


## Fractal Flame:
    -f, --flame                      If not provided (and -c not provided), a binary 
                                     fractal will be generated. If provided, the image  
                                     will be generated using the Fractal Flame flmae
                                     algorithm. Unfortunately, this mechanic is not 
                                     working well yet. Setting BS <= 4 might result in 
                                     better results.

    -g, --gamma GAMMA                Gamma parameter for the fractal frame algorithm.
                                     Default value is 20.

    -su, --sup SUP                   Supersampling ratio for the fractal flame algorithm. 
                                     Default value is 3.


<a  id="contact"></a>
Contact
===
The author of this repo can be mailed at [dsvyezhentsev@gmail.com](mailto:dsvyezhentsev@gmail.com).

<a  id="citations"></a>
Citations
===

```bibtex
@inproceedings{Barnsley2003ERGODICT,
  title={ERGODIC THEORY , FRACTAL TOPS AND COLOUR},
  author={Michael F. Barnsley},
  year={2003}
}

```


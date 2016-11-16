# pix2pix
Generating pixels from pixels

## Setup
Create folder '/path/to/data/' with subfolders 'train' and 'val'. Each subfolder contains pairs of images A and B.

Images should be named '1_A.jpg', '1_B.jpg', '2_A.jpg', etc.

Numbers must be sequential. Images must be jpgs. '1_A.jpg' must correspond to '1_B.jpg'.

## Display UI
Optionally, for displaying images during training and test, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

By default, the server listens on localhost. Pass `0.0.0.0` to allow external connections on any interface:
```bash
    th -ldisplay.start 8000 0.0.0.0
```
Then open `http://(hostname):(port)/` in your browser to load the remote desktop.
## Train
	DATA_ROOT=/path/to/data/ name=expt_name which_direction=AtoB th train.lua

Switch 'AtoB' to 'BtoA' to train translation in opposite direction.

Models are cached to './checkpoints/expt_name'.

## Test
	DATA_ROOT=/path/to/data/ name=expt_name which_direction=AtoB which_img=1 batchSize=10 th test.lua

This will run the model named 'expt_name' in direction 'AtoB' on images starting from 'which_img' to 'which_img'+'batch_size'-1.

Results, and a webpage to view them, are saved to './Results/expt_name'.

## Acknowledgments
Code borrows heavily from https://github.com/soumith/dcgan.torch

Additional code credits:
http://www.cs.virginia.edu/~vicente/urban/index.html


-- usage cityscapes: DATA_ROOT=/data/efros/isola/cityscapes/data_for_GAN_AE/ name=experiment_train2_cityscapes_anno2photo_latest which_direction=BtoA batchSize=8 how_many=96 which_img=1 th test.lua
-- usage facades: DATA_ROOT=/data/efros/isola/facades/data_for_GAN_AE/ name=experiment1_facades_anno2photo_latest which_direction=BtoA batchSize=8 how_many=96 which_img=1 aspect_ratio=1 th test.lua
-- usage image compression: DATA_ROOT=/data/efros/isola/image_compression/data_for_GAN_AE/ name=experiment1_image_compression_anno2photo_latest which_direction=AtoB batchSize=8 how_many=96 which_img=1 aspect_ratio=1 th test.lua
-- usage colorization: DATA_ROOT=/data/efros/isola/colorization/data_for_GAN_AE/ which_direction=AtoB name=experiment1_colorization_anno2photo_latest batchSize=25 how_many=100 loadSize=64 fineSize=64  which_img=1 aspect_ratio=1 th test.lua
-- usage BSDS: DATA_ROOT=/data/efros/isola/BSDS/data_for_GAN_AE/ which_direction=AtoB name=experiment_random_zs_BSDS_anno2photo_latest batchSize=8 how_many=16 which_img=1 aspect_ratio=1 th test.lua

require 'image'
require 'nn'
require 'ConcatNoise'
require 'Sampler'
require 'AddNoise'
require 'nngraph'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1,          -- # images in batch
    loadSize = 256,         -- scale images to this size
    fineSize = 256,         --  then crop to this size
    flip=0,                 -- horizontal mirroring data augmentation
    display = 1,            -- display samples while training. 0 = false
    display_id = 200,       -- display window id.
    gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    which_img = '1',        -- which test image to start with
    how_many = 'all',           -- how many test images to run 
    which_direction = 'BtoA',
    VAE = false,
    phase = 'val',
	  preprocess = 'regular',
    aspect_ratio = 1.0,        -- aspect ratio
    name = '',
    input_nc = 3, 
    output_nc = 3,
    serial_batches = 0,        -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,     -- iter into serial image list
    preload_data = false,
    checkpoints_dir = '/data/efros/isola/pix2pix/checkpoints',
    results_dir='/data/efros/isola/pix2pix/results/',
    cudnn = 1, -- set to 0 to not use cudnn
    which_epoch = 'latest',
}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'

local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- translation direction
local idx_A = nil
local idx_B = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc
if opt.which_direction=='AtoB' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  error(string.format('bad direction %s',opt.which_direction))
end
----------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
--netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
--netG:evaluate()

print(netG)

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
--util.optimizeInferenceMemory(netG) -- errors when netG is called more than once...


function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

if opt.how_many=='all' then
	opt.how_many=data:size()
end
opt.how_many=math.min(opt.how_many, data:size())

local filepaths = {} -- paths to images tested on
for n=1,math.floor(opt.how_many/opt.batchSize) do
	print('processing batch ' .. n)
	
	local data_curr, filepaths_curr = data:getBatch()
	filepaths_curr = util.basename_batch(filepaths_curr)
	print('filepaths_curr: ', filepaths_curr)

	input = data_curr[{ {}, idx_A, {}, {} }]
	target = data_curr[{ {}, idx_B, {}, {} }]
	
	if opt.gpu > 0 then
	    input = input:cuda()
	end
--	   if opt.preprocess == 'colorization' then 
--      local real_A_s = util.scaleBatch(real_A:float(),100,100)
--      local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
--      local real_B_s = util.scaleBatch(real_B:float(),100,100)
--      disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input', normalize=false})
--      disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output', normalize=false})
--      disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target', normalize=false})
--    else
--      disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input', normalize=false})
--      disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output', normalize=false})
--      disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target', normalize=false})
	
	if opt.preprocess == 'colorization' then
--	   print('input', input:size())
	   local output_AB = netG:forward(input):float()
	   local input_L = input:float() 
--	   print('output_AB', output_AB:size())
	   output = util.deprocessLAB_batch(input_L, output_AB)
--	   print('output', output:size())
--	   os.exit()
--     output = output:float()
     local target_AB = target:float()
     target = util.deprocessLAB_batch(input_L, target_AB)
     input = util.deprocessL_batch(input_L)
--     print('input', input:min(), input:max())
--     print('target', target:min(), target:max())
--     print('output', output:min(), output:max())
    
	else 
  	output = util.deprocess_batch(netG:forward(input))
  	input = util.deprocess_batch(input):float()
  	output = output:float()
  	target = util.deprocess_batch(target):float()
	end
	paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
	local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
	paths.mkdir(image_dir)
	paths.mkdir(paths.concat(image_dir,'input'))
	paths.mkdir(paths.concat(image_dir,'output'))
	paths.mkdir(paths.concat(image_dir,'target'))
	print(input:size())
	print(output:size())
	print(target:size())
	for i=1, opt.batchSize do
		image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
		image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
		image.save(paths.concat(image_dir,'target',filepaths_curr[i]), image.scale(target[i],target[i]:size(2),target[i]:size(3)/opt.aspect_ratio))
	end
	print('Saved images to: ', image_dir)
	
	if opt.display then
	  if opt.preprocess == 'regular' then
	    disp = require 'display'
--	  print(input)
--	  os.exit()
		disp.image(util.scaleBatch(input,100,100),{win=opt.display_id, title='input', normalize=false})
		disp.image(util.scaleBatch(output,100,100),{win=opt.display_id+1, title='output', normalize=false})
		disp.image(util.scaleBatch(target,100,100),{win=opt.display_id+2, title='target', normalize=false})
		
	    print('Displayed images')
	  end
	end
	
	filepaths = TableConcat(filepaths, filepaths_curr)
end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))

io.write('<table style="text-align:center;">')

io.write('<tr><td>Image #</td><td>Input</td><td>Output</td><td>Ground Truth</td></tr>')
for i=1, #filepaths do
	io.write('<tr>')
	io.write('<td>' .. filepaths[i] .. '</td>')
	io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
	io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
	io.write('<td><img src="./images/target/' .. filepaths[i] .. '"/></td>')
	io.write('</tr>')
end

io.write('</table>')
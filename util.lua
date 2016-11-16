local util = {}

--local mat = require 'matio'
require 'torch'

function util.normalize(img)
  -- rescale image to 0 .. 1
  local min = img:min()
  local max = img:max()
  
  img = torch.FloatTensor(img:size()):copy(img)
  img:add(-min):mul(1/(max-min))
  return img
end

function util.normalizeBatch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.normalize(batch[i]:squeeze())
	end
	return batch
end

function util.basename_batch(batch)
	for i = 1, #batch do
		batch[i] = paths.basename(batch[i])
	end
	return batch
end



-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
-- modified from: https://github.com/jcjohnson/neural-style/blob/master/neural_style.lua
function util.preprocess(img)
  --local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
--  print(img:size())
  img = img:index(1, perm)--:mul(256.0)
  --mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  --img:add(-1, mean_pixel)
--   print(img:size())
  img = img:mul(2):add(-1)
--   print(img:size())
  assert(img:max()<=1,"badly scaled inputs")
  assert(img:min()>=-1,"badly scaled inputs")
  
  return img
end

-- Undo the above preprocessing.
-- modified from: https://github.com/jcjohnson/neural-style/blob/master/neural_style.lua
function util.deprocess(img)
  --local mean_pixel = torch.FloatTensor({103.939, 116.779, 123.68})
  --mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  --img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm)--:div(256.0)
  
  img = img:add(1):div(2)
  
  return img
end

function util.deprocessLAB(L, AB)
    local L2 = torch.Tensor(L:size()):copy(L)
    if L2:dim() == 3 then
      L2 = L2[{1, {}, {} }]
    end
    local AB2 = torch.Tensor(AB:size()):copy(AB)
    AB2 = torch.clamp(AB2, -1.0, 1.0)
--    local AB2 = AB
    L2 = L2:add(1):mul(50.0)
    AB2 = AB2:mul(110.0)
--    print('before deprocessLAB')
--    print(L2:size(), L2:min(), L2:max())
--    print(AB2:size(), AB2:min(), AB2:max())
    
    L2 = L2:reshape(1, L2:size(1), L2:size(2))
--    print('after deprocessLAB')
--    print(L2:size(), L2:min(), L2:max())
--    print(AB2:size(), AB2:min(), AB2:max())
    im_lab = torch.cat(L2, AB2, 1)
    im_rgb = torch.clamp(image.lab2rgb(im_lab):mul(255.0), 0.0, 255.0)/255.0
--    local perm = torch.LongTensor{3, 2, 1}
--    im_rgb = im_rgb:index(1, perm)--:mul(256.0): brg, rgb
--    print('image',im_rgb:size(),im_rgb:min(), im_rgb:max())
    return im_rgb
end

function util.deprocessL(L)
    local L2 = torch.Tensor(L:size()):copy(L)
    L2 = L2:add(1):mul(255.0/2.0)
--    print('before deprocess L')
--    print(L2:size(), L2:min(), L2:max())
    if L2:dim()==2 then
      L2 = L2:reshape(1,L2:size(1),L2:size(2))
    end
    L2 = L2:repeatTensor(L2,3,1,1)/255.0
    
--    print('after deprocess L')
--    print(L2:size(), L2:min(), L2:max())
    return L2
end

function util.deprocessL_batch(batch)
  local batch_new = {}
  for i = 1, batch:size(1) do
    batch_new[i] = util.deprocessL(batch[i]:squeeze())
  end
  return batch_new
end

function util.deprocessLAB_batch(batchL, batchAB)
  local batch = {}
--  print('batchL', batchL:size())
--  print('batchAB', batchAB:size())
--  print('batch_sz', batchL:size(1))
  for i = 1, batchL:size(1) do
    batch[i] = util.deprocessLAB(batchL[i]:squeeze(), batchAB[i]:squeeze())
  end
--  print(batch)
--  print('batch', batch:size())
--  os.exit()
  return batch
end

function util.preprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.preprocess(batch[i]:squeeze())
	end
	return batch
end

function util.deprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.deprocess(batch[i]:squeeze())
	end
	return batch
end


function util.scaleBatch(batch,s1,s2)
	local scaled_batch = torch.Tensor(batch:size(1),batch:size(2),s1,s2)
	for i = 1, batch:size(1) do
		scaled_batch[i] = image.scale(batch[i],s1,s2):squeeze()
	end
	return scaled_batch
end


function util.toTrivialBatch(input)
	return input:reshape(1,input:size(1),input:size(2),input:size(3))
end
function util.fromTrivialBatch(input)
	return input[1]
end



function util.scaleImage(input, loadSize)
	
	-- replicate bw images to 3 channels
	if input:size(1)==1 then
		input = torch.repeatTensor(input,3,1,1)
	end
	
	--[[
    -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
    local iW = input:size(3)
    local iH = input:size(2)
    if iW < iH then
       input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
    else
       input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
    end
	--]]
	
	input = image.scale(input, loadSize, loadSize)
	
    return input
end

function util.getAspectRatio(path)
	local input = image.load(path, 3, 'float')
	local ar = input:size(3)/input:size(2)
	return ar
end

function util.loadImage(path, loadSize, nc)
--   print(path)
   local input = image.load(path, 3, 'float')

--   print(input:size())
   input= util.preprocess(util.scaleImage(input, loadSize))
  
--   print('nc=' .. nc)
   if nc == 1 then
    input = input[{{1}, {}, {}}]
   end
--   print(input:size())
   return input 
end



--[[
function loadMat(path)
   
   --local inputFile = hdf5.open(path, 'r')
   --local input = inputFile:read('/data'):all()
   --inputFile:close()
   
   local input = mat.load(path,'data'):permute(3,1,2,4):double():div(255)
   
   local a = scaleImage(input[{ {}, {}, {}, {1} }]:squeeze())
   local b = scaleImage(input[{ {}, {}, {}, {2} }]:squeeze())
   input = torch.cat(a,b,4)
   
   return input
end
--]]

function util.save_pairwise(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
	-- TODO: figure out why giant storage-offsets being created on typecast
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output = netsave.output.new()
    netsave.gradInput = netsave.gradInput.new()

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.load_pairwise(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
   return net
end

function util.save(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
	-- TODO: figure out why giant storage-offsets being created on typecast
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output = netsave.output.new()
    netsave.gradInput = netsave.gradInput.new()

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.save_large(filename, net, gpu)
	
	net:clearState() -- clears state variables like output, gradInput, etc
	
    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end
	
    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
					      l.kW, l.kH, l.dW, l.dH, 
					      l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
					       l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end
	end
	
	torch.save(filename, netsave)
end

function util.load(filename, opt)
	if opt.cudnn then
		require 'cudnn'
	end
	local net = torch.load(filename)
	if opt.gpu > 0 then
		require 'cunn'
		net:cuda()
		
		-- calling cuda on cudnn saved nngraphs doesn't change all variables to cuda, so do it below
		if net.forwardnodes then
			for i=1,#net.forwardnodes do
				if net.forwardnodes[i].data.module then
					net.forwardnodes[i].data.module:cuda()
				end
			end
		end
		
	else
		net:float()
	end
	net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
	return net
end

function util.cudnn(net)
	require 'cudnn'
	require 'cudnn_convert_custom'
	return cudnn_convert_custom(net, cudnn)
end

-- a function to do memory optimizations by 
-- setting up double-buffering across the network.
-- this drastically reduces the memory needed to generate samples.
function util.optimizeInferenceMemory(net)
    local finput, output, outputB
    net:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end

return util

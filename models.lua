require 'nngraph'

function defineG_encoder_decoder(input_nc, output_nc, ngf, nz)
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end

function defineG_encoder_decoder_v2(input_nc, output_nc, ngf, nz)
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, nz, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(nz)
	-- input is (ngf * 8) x 1 x 1
	
	z = e8 - nn.Tanh() - nn.Dropout(0.5)
	-- bottleneck is ngf*8
	
	d1 = z - nn.SpatialFullConvolution(nz, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end
	
function defineG_encoder_decoder_v1(input_nc, output_nc, ngf, nz, n_layers)

    local netG_encoder = nn.Sequential()
    -- input is (input_nc) x 256 x 256
    netG_encoder:add(nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1))
    netG_encoder:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netG_encoder:add(nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG_encoder:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netG_encoder:add(nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG_encoder:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netG_encoder:add(nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
    netG_encoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 16 x 16
    for n = 1, n_layers do
      netG_encoder:add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1))
      netG_encoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
    end
    -- state size: (ndf*8) x 2 x 2
    netG_encoder:add(nn.SpatialConvolution(ngf * 8, nz, 4, 4, 2, 2, 1, 1))
    --netG_encoder:add(SpatialBatchNormalization(nz)):add(nn.LeakyReLU(0.2, true))
    netG_encoder:add(nn.SpatialBatchNormalization(nz)):add(nn.Tanh()) -- outputs in range [-1,1]
    -- state size Z: nz x 1 x 1
   
   
    local netG_decoder = nn.Sequential()
    --local N_noise_dims = 100
    --netG:add(nn.ConcatNoise(N_noise_dims))
    --netG:add(nn.Sampler()) -- adds noise to latent representation
    netG_decoder:add(nn.Dropout(0.5))
    -- input is Z, going into a convolution
    netG_decoder:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4, 2, 2, 1, 1))
    netG_decoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    
    for n = 1, n_layers do 
      netG_decoder:add(nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1))
      netG_decoder:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
    end
    
	-- state size: (ngf*8) x 16 x 16
    netG_decoder:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG_decoder:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 32 x 32
    netG_decoder:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    netG_decoder:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
    -- state size: (ngf*2) x 64 x 64
    netG_decoder:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    netG_decoder:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    -- state size: (ngf) x 128 x 128
    netG_decoder:add(nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1))
    netG_decoder:add(nn.Tanh())
    -- state size: (output_nc) x 256 x 256
   
    local netG = nn.Sequential()
    netG:add(netG_encoder)
    netG:add(netG_decoder)
	
	return netG
end

function defineG_hypercolumns(input_nc, output_nc, ngf)
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 2) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 2) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 2) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 2) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 2) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 2) x 1 x 1
	
	u1 = e1 - nn.SpatialUpSamplingBilinear(2)
	u2 = e2 - nn.SpatialUpSamplingBilinear(4)
	u3 = e3 - nn.SpatialUpSamplingBilinear(8)
	u4 = e4 - nn.SpatialUpSamplingBilinear(16)
	u5 = e5 - nn.SpatialUpSamplingBilinear(32)
	u6 = e6 - nn.SpatialUpSamplingBilinear(64)
	u7 = e7 - nn.SpatialUpSamplingBilinear(128)
	u8 = e8 - nn.SpatialUpSamplingBilinear(256)
	
	c = {u1,u2,u3,u4,u5,u6,u7,u8} - nn.JoinTable(2)
	d1 = c - nn.LeakyReLU(0.2, true) - nn.Linear(ngf*47,ngf*8) - nn.BatchNormalization(ngf * 8)
	d2 = d1 - nn.LeakyReLU(0.2, true) - nn.Linear(ngf*8,ngf*8) - nn.BatchNormalization(ngf * 8)
	d3 = d2 - nn.LeakyReLU(0.2, true) - nn.Linear(ngf*8,ngf*8) - nn.BatchNormalization(ngf * 8)
	d4 = d3 - nn.LeakyReLU(0.2, true) - nn.Linear(ngf*8,output_nc)
	
	o1 = d4 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end

function defineG_unet(input_nc, output_nc, ngf)
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	d1 = {d1_,e7} - nn.JoinTable(2)
	d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	d2 = {d2_,e6} - nn.JoinTable(2)
	d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	d3 = {d3_,e5} - nn.JoinTable(2)
	d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d4 = {d4_,e4} - nn.JoinTable(2)
	d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d5 = {d5_,e3} - nn.JoinTable(2)
	d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d6 = {d6_,e2} - nn.JoinTable(2)
	d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d7 = {d7_,e1} - nn.JoinTable(2)
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end


function defineG_unet2(input_nc, output_nc, ngf)
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	d1 = {d1_,e7} - nn.JoinTable(2)
	d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	d2 = {d2_,e6} - nn.JoinTable(2)
	d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	d3 = {d3_,e5} - nn.JoinTable(2)
	d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d4 = {d4_,e4} - nn.JoinTable(2)
	d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d5 = {d5_,e3} - nn.JoinTable(2)
	d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d6 = {d6_,e2} - nn.JoinTable(2)
	d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d7 = {d7_,e1} - nn.JoinTable(2)
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (nc) x 256 x 256
	
	-- the idea of d9 and d10 is to clean up artifacts from the upsampling procedure
	d9 = d8 - nn.ReLU(true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
	d10 = d9 - nn.ReLU(true) - nn.SpatialConvolution(ngf, output_nc, 3, 3, 1, 1, 1, 1) 
	
	o1 = d10 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end


function defineG_unet_upsample(input_nc, output_nc, ngf)
	
	--local UpSample = nn.SpatialUpSamplingNearest -- note: with this setting, need to change padding or kernel sizes of conv filters after UpSample
	local UpSample = nn.SpatialUpSamplingBilinear
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1_ = e8 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 2 x 2
	d1 = {d1_,e7} - nn.JoinTable(2)
	d2_ = d1 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 4 x 4
	d2 = {d2_,e6} - nn.JoinTable(2)
	d3_ = d2 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
	-- input is (ngf * 8) x 8 x 8
	d3 = {d3_,e5} - nn.JoinTable(2)
	d4_ = d3 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d4 = {d4_,e4} - nn.JoinTable(2)
	d5_ = d4 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d5 = {d5_,e3} - nn.JoinTable(2)
	d6_ = d5 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d6 = {d6_,e2} - nn.JoinTable(2)
	d7_ = d6 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 2 * 2, ngf, 4, 4, 1, 1, 2, 2) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d7 = {d7_,e1} - nn.JoinTable(2)
	d8 = d7 - nn.ReLU(true) - UpSample(2) - nn.SpatialConvolution(ngf * 2, output_nc, 4, 4, 1, 1, 2, 2)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
end


function defineG_unet_super_stochastic(input_nc, output_nc, ngf)
	
	-- input is (nc) x 256 x 256
	require 'AddNoise'
	e0 = - nn.AddNoise(input_nc)
	
	-- input is (nc) x 256 x 256
	e1 = e0 - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	
	d5_ = e4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d5 = {d5_,e3} - nn.JoinTable(2)
	d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d6 = {d6_,e2} - nn.JoinTable(2)
	d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d7 = {d7_,e1} - nn.JoinTable(2)
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e0},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
	
	
	--[[
	--require 'NestedDropout'
	--local Dropout = nn.NestedDropout
	local Dropout = nn.Dropout
	
	p = 0.5
	--ngf = ngf*(1/p) -- * so that p*ngf = intended ngf
	
	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1_ = e8 - Dropout(p) - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 2 x 2
	d1 = {d1_,e7} - nn.JoinTable(2) - Dropout(p)
	d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 4 x 4
	d2 = {d2_,e6} - nn.JoinTable(2) - Dropout(p)
	d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 8 x 8
	d3 = {d3_,e5} - nn.JoinTable(2) - Dropout(p)
	d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
	-- input is (ngf * 8) x 16 x 16
	d4 = {d4_,e4} - nn.JoinTable(2) - Dropout(p)
	d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
	-- input is (ngf * 4) x 32 x 32
	d5 = {d5_,e3} - nn.JoinTable(2) - Dropout(p)
	d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
	-- input is (ngf * 2) x 64 x 64
	d6 = {d6_,e2} - nn.JoinTable(2) - Dropout(p)
	d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
	-- input is (ngf) x128 x 128
	d7 = {d7_,e1} - nn.JoinTable(2) - Dropout(p)
	d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
	-- input is (nc) x 256 x 256
	
	o1 = d8 - nn.Tanh()
	
	netG = nn.gModule({e1},{o1})
	
	--graph.dot(netG.fg,'netG')
	
	return netG
	--]]
end



function defineD_unet(input_nc, output_nc, ndf)
	local netD = nn.Sequential()

	-- input is (nc) x 256 x 256
	e1 = - nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1)
	-- input is (ngf) x 128 x 128
	e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 2)
	-- input is (ngf * 2) x 64 x 64
	e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 4)
	-- input is (ngf * 4) x 32 x 32
	e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 16 x 16
	e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 8 x 8
	e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 4 x 4
	e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 1 x 1
	
	d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 8, ndf * 8, 4, 4) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 4 x 4
	d1 = {d1_,e6} - nn.JoinTable(2)
	d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 8 * 2, ndf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 8 x 8
	d2 = {d2_,e5} - nn.JoinTable(2)
	d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 8 * 2, ndf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 8)
	-- input is (ngf * 8) x 16 x 16
	d3 = {d3_,e4} - nn.JoinTable(2)
	d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 8 * 2, ndf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 4)
	-- input is (ngf * 8) x 32 x 32
	d4 = {d4_,e3} - nn.JoinTable(2)
	d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 4 * 2, ndf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf * 2)
	-- input is (ngf * 4) x 64 x 64
	d5 = {d5_,e2} - nn.JoinTable(2)
	d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ndf * 2 * 2, ndf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ndf)
	-- input is (ngf * 2) x 128 x 128
	d6 = {d6_,e1} - nn.JoinTable(2)
	d7 = d6 - nn.ReLU(true) - nn.SpatialConvolution(ndf * 1 * 2, 1, 4, 4, 1, 1, 0, 0)
	
	o1 = d7 - nn.Sigmoid()
	
	netD = nn.gModule({e1},{o1})
   
	return netD
end

function defineD_basic_dilated(input_nc, output_nc, ndf)
	local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialDilatedConvolution(ndf, ndf * 2, 3, 3, 1, 1, 2, 2, 2, 2))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 128 x 128
    netD:add(nn.SpatialDilatedConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 4, 4, 4, 4))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 128 x 128
    netD:add(nn.SpatialDilatedConvolution(ndf * 2, ndf * 2, 3, 3, 1, 1, 8, 8, 8, 8))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 4, 4, 1, 1, 1, 1))
    -- state size: 1 x 128 x 128
   
    netD:add(nn.Sigmoid())
    -- state size: 1 x 128 x 128
	
	return netD
end


function defineD_basic_v1(input_nc, output_nc, ndf)
	local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netD:add(nn.SpatialConvolution(ndf * 2, ndf*4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 1, 1, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 31 x 31
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4, 1, 1, 1, 1))
    -- state size: 1 x 30 x 30
   
    netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
	
	return netD
end

function defineD_basic(input_nc, output_nc, ndf)
    
	n_layers = 3
	return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

-- rf=1
function defineD_pixelGAN(input_nc, output_nc, ndf)
	
	local netD = nn.Sequential()
	
	-- input is (nc) x 256 x 256
	netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf) x 256 x 256
	netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
	netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
	-- state size: (ndf*8) x 31 x 31
	netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
	-- state size: 1 x 30 x 30
	
	netD:add(nn.Sigmoid())
	-- state size: 1 x 30 x 30
		
	return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
	
	if n_layers==0 then
		return defineD_pixelGAN(input_nc, output_nc, ndf)
	else
	
		local netD = nn.Sequential()
		
    	-- input is (nc) x 256 x 256
    	netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    	netD:add(nn.LeakyReLU(0.2, true))
		
		nf_mult = 1
    	for n = 1, n_layers-1 do 
			nf_mult_prev = nf_mult
			nf_mult = math.min(2^n,8)
			netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
			netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    	end
		
    	-- state size: (ndf*4) x 32 x 32
		nf_mult_prev = nf_mult
		nf_mult = math.min(2^n_layers,8)
    	netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
    	netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    	-- state size: (ndf*8) x 31 x 31
    	netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
    	-- state size: 1 x 30 x 30
    	
    	netD:add(nn.Sigmoid())
    	-- state size: 1 x 30 x 30
			
		return netD
	end
end


-- rf is 4*2^(n_layers-1)?
-- if n_layers==0 then rf is 1x1 (PixelGAN)
function defineD_n_layers_v1(input_nc, output_nc, ndf, n_layers)
	
	local netD = nn.Sequential()
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
	
	nf_mult = 1
    for n = 1, n_layers do 
		nf_mult_prev = nf_mult
		nf_mult = math.min(2^n,8)
		netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
		netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    end
	
	netD:add(nn.SpatialConvolution(ndf * nf_mult, ndf * nf_mult, 1, 1, 1, 1, 0, 0)) 
	netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
    netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 1, 1, 1, 1, 0, 0))   
    netD:add(nn.Sigmoid())
	
	return netD
end


function defineD_smallpatchGAN(input_nc, output_nc, ndf)
    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 1, 1, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 127 x 127
    netD:add(nn.SpatialConvolution(ndf * 2, ndf * 2, 4, 4, 1, 1, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 126 x 126
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 4, 4, 1, 1, 1, 1))
    -- state size: 1 x 125 x 125
   
    netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
	
	return netD
end

function defineD_imageGAN(input_nc, output_nc, ndf)
    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netD:add(nn.SpatialConvolution(ndf * 2, ndf*4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 16 x 16
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 8 x 8
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 2 x 2
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4, 2, 2, 1, 1))
    -- state size: 1 x 1 x 1
   
    netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
	
	return netD
end
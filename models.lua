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
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    
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
        
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
        -- state size: 1 x (N-2) x (N-2)
        
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
    end
end
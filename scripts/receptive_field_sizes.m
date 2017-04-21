% modified from: https://github.com/rbgirshick/rcnn/blob/master/utils/receptive_field_sizes.m
% 
% RCNN LICENSE:
%
% Copyright (c) 2014, The Regents of the University of California (Regents)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met: 
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer. 
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution. 
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function receptive_field_sizes()


% compute input size from a given output size
f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize;


%% n=1 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(1, 4, 1), ...   % conv2 -> conv3
             4, 1), ...   % conv1 -> conv2
             4, 2);       % input -> conv1

fprintf('n=1 discriminator receptive field size: %d\n', out);


%% n=2 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(1, 4, 1), ...   % conv3 -> conv4
             4, 1), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1

fprintf('n=2 discriminator receptive field size: %d\n', out);


%% n=3 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(f(1, 4, 1), ...   % conv4 -> conv5
             4, 1), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1

fprintf('n=3 discriminator receptive field size: %d\n', out);


%% n=4 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(f(f(1, 4, 1), ...   % conv5 -> conv6
             4, 1), ...   % conv4 -> conv5
             4, 2), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1

fprintf('n=4 discriminator receptive field size: %d\n', out);


%% n=5 discriminator

% fix the output size to 1 and derive the receptive field in the input
out = ...
f(f(f(f(f(f(f(1, 4, 1), ...   % conv6 -> conv7
             4, 1), ...   % conv5 -> conv6
             4, 2), ...   % conv4 -> conv5
             4, 2), ...   % conv3 -> conv4
             4, 2), ...   % conv2 -> conv3
             4, 2), ...   % conv1 -> conv2
             4, 2);       % input -> conv1

fprintf('n=5 discriminator receptive field size: %d\n', out);
import torch
from torch.nn import functional as F
from torch import nn
from util import *
import tensorboardX


class MBConv(nn.Module):
    """
    
    Mobile Inverted Residual Bottleneck Block
    
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self,block_args,global_params):
        super().__init__()
        self.block_args=block_args
        self.bn_mom = 1-global_params.bn_mom
        self.bn_eps = global_params.bn_eps
        self.has_se = (self.block_args.se_ratio is not None) and (0<self.block_args.se_ratio<=1)
        self.id_skip = block_args.id_skip
        Conv2D = Conv2dSamePadding

        inpt = self.block_args.input_filters
        out = self.block_args.input_filters*self.block_args.expand_ratio
        
        # Expansion
        if self.block_args.expand_ratio!=1:
            self.expand_conv = Conv2D(inpt,out,kernel_size = 1, bias = False)
            self.bn0 = nn.BatchNorm2d(num_features = out,momentum = self.bn_mom, eps = self.bn_eps)
        
        # Depthwise
        k = self.block_args.kernel_size
        s = self.block_args.stride
        self.depthwise_conv = Conv2D(out,out,groups=out,
                                    kernel_size=k,stride=s,bias = False)
        self.bn1 = nn.BatchNorm2d(num_features=out,momentum = self.bn_mom, eps = self.bn_eps)

        # Squeeze and Excitation
        if self.has_se:
            num_squeeze_channels = max(1,int(self.block_args.input_filters*self.block_args.se_ratio))
            self.se_reduce = Conv2D(out,num_squeeze_channels,kernel_size=1)
            self.se_expand = Conv2D(num_squeeze_channels,out,kernel_size=1)

        # Output 
        final_outp = self.block_args.output_filters
        self.project_conv = Conv2D(out, final_outp,kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(final_outp,momentum = self.bn_mom, eps = self.bn_eps)
        self.swish = MemoryEfficientSwish()

    def forward(self,inputs,drop_connect_rate = None):
        x = inputs
        if self.block_args.expand_ratio!=1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        if self.has_se:
            x_squeez = F.adaptive_avg_pool2d(x,1)
            x_squeez = self.se_reduce(x_squeez)
            x_squeez = self.swish(x_squeez)
            x_squeez = self.se_expand(x_squeez)
            x = torch.sigmoid(x_squeez)*x
        x = self.project_conv(x)
        x = self.bn2(x)
        
        inpt_filters = self.block_args.input_filters
        outpt_filters = self.block_args.output_filters
        if self.id_skip and self.block_args.stride == 1 and inpt_filters == outpt_filters:
            if drop_connect_rate:
                x = drop_connect(x,p=drop_connect_rate,training = self.training)
            x = x+inputs
        return x    


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.global_params = global_params
        self.blocks_args = blocks_args            
        Conv2D = Conv2dSamePadding
        bn_mom = global_params.bn_mom
        bn_eps = global_params.bn_eps
        in_channels = 3 
        out_channels = round_filters(32,self.global_params)
        self.conv_stem = Conv2D(in_channels, out_channels,kernel_size = 3, stride=2,bias = False)
        self.bn0 = nn.BatchNorm2d(out_channels,bn_mom,bn_eps)
        self.blocks = nn.ModuleList([])
        for block_args in self.blocks_args:
            block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, self.global_params),
            output_filters=round_filters(block_args.output_filters, self.global_params),
            num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )
            self.blocks.append(MBConv(block_args,global_params))
            if block_args.num_repeat>1:
                block_args = block_args._replace(input_filters=block_args.output_filters,stride=1)
            for _ in range(block_args.num_repeat -1):
                self.blocks.append(MBConv(block_args,global_params))
        # head
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self.global_params)
        self.conv_head = Conv2D(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels,bn_mom,bn_eps)
        # final
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.global_params.dropout_rate)
        self.fc = nn.Linear(out_channels,self.global_params.num_classes)
        self.swish = MemoryEfficientSwish()
    
    
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self.blocks:
            block.set_swish(memory_efficient)
        
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self.swish(self.bn0(self.conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
                #print(block)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self.swish(self.bn1(self.conv_head(x)))
        return x
        
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  
    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)
    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=True, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        if load_weights:
            load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = Conv2dSamePadding
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res
    
    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models)) 



if __name__=="__main__":
    compound_coeff = 0
    model = EfficientNet.from_name(f"efficientnet-b{compound_coeff}")
    #print(model)
    
    dummy = torch.ones([1,3,224,224])
    pred = model(dummy)
    summary = tensorboardX.SummaryWriter(logdir = './logs',comment='Effnet_Summary')
    summary.add_graph(model,dummy)
    summary.close()
    print('done')

!Ö 
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
h
BatchMatMul
x"T
y"T
output"T"
Ttype:
	2"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
ì
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ë

DecodeJpeg
contents	
image"
channelsint "
ratioint"
fancy_upscalingbool("!
try_recover_truncatedbool( "#
acceptable_fractionfloat%  ?"

dct_methodstring 
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
í
ParseSingleExample

serialized
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense"

num_sparseint("
sparse_keyslist(string)("

dense_keyslist(string)("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.12.02v1.12.0-0-ga6d8ffae098

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
shape: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
U
input_example_tensorPlaceholder*
dtype0*
_output_shapes
: *
shape: 
f
%ParseSingleExample/key_content_imagesConst*
valueB B *
dtype0*
_output_shapes
: 
c
 ParseSingleExample/Reshape/shapeConst*
dtype0*
_output_shapes
: *
valueB 

ParseSingleExample/ReshapeReshape%ParseSingleExample/key_content_images ParseSingleExample/Reshape/shape*
_output_shapes
: *
T0
d
#ParseSingleExample/key_style_imagesConst*
valueB B *
dtype0*
_output_shapes
: 
e
"ParseSingleExample/Reshape_1/shapeConst*
valueB *
dtype0*
_output_shapes
: 

ParseSingleExample/Reshape_1Reshape#ParseSingleExample/key_style_images"ParseSingleExample/Reshape_1/shape*
T0*
_output_shapes
: 
¸
%ParseSingleExample/ParseSingleExampleParseSingleExampleinput_example_tensorParseSingleExample/ReshapeParseSingleExample/Reshape_1*
sparse_types
 *
dense_shapes
: : *
sparse_keys
 *
Tdense
2*

num_sparse *.

dense_keys 
content_imagesstyle_images*
_output_shapes
: : 


DecodeJpeg
DecodeJpeg%ParseSingleExample/ParseSingleExample*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
^
resize_images/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0

resize_images/ExpandDims
ExpandDims
DecodeJpegresize_images/ExpandDims/dim*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
c
resize_images/sizeConst*
valueB"      *
dtype0*
_output_shapes
:

resize_images/ResizeBilinearResizeBilinearresize_images/ExpandDimsresize_images/size*(
_output_shapes
:*
T0

resize_images/SqueezeSqueezeresize_images/ResizeBilinear*
squeeze_dims
 *
T0*$
_output_shapes
:
P
ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
r

ExpandDims
ExpandDimsresize_images/SqueezeExpandDims/dim*
T0*(
_output_shapes
:

DecodeJpeg_1
DecodeJpeg'ParseSingleExample/ParseSingleExample:1*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
channels
`
resize_images_1/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0

resize_images_1/ExpandDims
ExpandDimsDecodeJpeg_1resize_images_1/ExpandDims/dim*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
e
resize_images_1/sizeConst*
valueB"      *
dtype0*
_output_shapes
:

resize_images_1/ResizeBilinearResizeBilinearresize_images_1/ExpandDimsresize_images_1/size*
T0*(
_output_shapes
:

resize_images_1/SqueezeSqueezeresize_images_1/ResizeBilinear*
squeeze_dims
 *
T0*$
_output_shapes
:
R
ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
x
ExpandDims_1
ExpandDimsresize_images_1/SqueezeExpandDims_1/dim*
T0*(
_output_shapes
:
^
ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Ñ
Bvgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   *4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel
»
@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *8JÌ½*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel
»
@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *8JÌ=*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel

Jvgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBvgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel
¢
@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/subSub@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/max@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*
_output_shapes
: 
¼
@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/mulMulJvgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/RandomUniform@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*&
_output_shapes
:@
®
<vgg19_encoder/block1_conv1/kernel/Initializer/random_uniformAdd@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/mul@vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform/min*&
_output_shapes
:@*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel
·
!vgg19_encoder/block1_conv1/kernel
VariableV2*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*
shape:@*
dtype0*&
_output_shapes
:@
ú
(vgg19_encoder/block1_conv1/kernel/AssignAssign!vgg19_encoder/block1_conv1/kernel<vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*&
_output_shapes
:@
¼
&vgg19_encoder/block1_conv1/kernel/readIdentity!vgg19_encoder/block1_conv1/kernel*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*&
_output_shapes
:@
²
1vgg19_encoder/block1_conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *2
_class(
&$loc:@vgg19_encoder/block1_conv1/bias

vgg19_encoder/block1_conv1/bias
VariableV2*2
_class(
&$loc:@vgg19_encoder/block1_conv1/bias*
shape:@*
dtype0*
_output_shapes
:@
Ý
&vgg19_encoder/block1_conv1/bias/AssignAssignvgg19_encoder/block1_conv1/bias1vgg19_encoder/block1_conv1/bias/Initializer/zeros*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv1/bias*
_output_shapes
:@
ª
$vgg19_encoder/block1_conv1/bias/readIdentityvgg19_encoder/block1_conv1/bias*
_output_shapes
:@*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv1/bias
y
(vgg19_encoder/block1_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
º
!vgg19_encoder/block1_conv1/Conv2DConv2D
ExpandDims&vgg19_encoder/block1_conv1/kernel/read*
strides
*
paddingSAME*(
_output_shapes
:@*
T0
©
"vgg19_encoder/block1_conv1/BiasAddBiasAdd!vgg19_encoder/block1_conv1/Conv2D$vgg19_encoder/block1_conv1/bias/read*(
_output_shapes
:@*
T0
~
vgg19_encoder/block1_conv1/ReluRelu"vgg19_encoder/block1_conv1/BiasAdd*
T0*(
_output_shapes
:@
Ñ
Bvgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*
dtype0*
_output_shapes
:
»
@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *:Í½*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*
dtype0*
_output_shapes
: 
»
@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:Í=*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*
dtype0*
_output_shapes
: 

Jvgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniformBvgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel
¢
@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/subSub@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/max@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*
_output_shapes
: 
¼
@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/mulMulJvgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/RandomUniform@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*&
_output_shapes
:@@
®
<vgg19_encoder/block1_conv2/kernel/Initializer/random_uniformAdd@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/mul@vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*&
_output_shapes
:@@
·
!vgg19_encoder/block1_conv2/kernel
VariableV2*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*
shape:@@*
dtype0*&
_output_shapes
:@@
ú
(vgg19_encoder/block1_conv2/kernel/AssignAssign!vgg19_encoder/block1_conv2/kernel<vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*&
_output_shapes
:@@
¼
&vgg19_encoder/block1_conv2/kernel/readIdentity!vgg19_encoder/block1_conv2/kernel*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel*&
_output_shapes
:@@*
T0
²
1vgg19_encoder/block1_conv2/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *2
_class(
&$loc:@vgg19_encoder/block1_conv2/bias*
dtype0

vgg19_encoder/block1_conv2/bias
VariableV2*
dtype0*
_output_shapes
:@*2
_class(
&$loc:@vgg19_encoder/block1_conv2/bias*
shape:@
Ý
&vgg19_encoder/block1_conv2/bias/AssignAssignvgg19_encoder/block1_conv2/bias1vgg19_encoder/block1_conv2/bias/Initializer/zeros*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv2/bias*
_output_shapes
:@
ª
$vgg19_encoder/block1_conv2/bias/readIdentityvgg19_encoder/block1_conv2/bias*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv2/bias*
_output_shapes
:@
y
(vgg19_encoder/block1_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ï
!vgg19_encoder/block1_conv2/Conv2DConv2Dvgg19_encoder/block1_conv1/Relu&vgg19_encoder/block1_conv2/kernel/read*
paddingSAME*(
_output_shapes
:@*
T0*
strides

©
"vgg19_encoder/block1_conv2/BiasAddBiasAdd!vgg19_encoder/block1_conv2/Conv2D$vgg19_encoder/block1_conv2/bias/read*
T0*(
_output_shapes
:@
~
vgg19_encoder/block1_conv2/ReluRelu"vgg19_encoder/block1_conv2/BiasAdd*
T0*(
_output_shapes
:@
³
!vgg19_encoder/block1_pool/MaxPoolMaxPoolvgg19_encoder/block1_conv2/Relu*
strides
*
ksize
*
paddingVALID*(
_output_shapes
:@
Ñ
Bvgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @      *4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*
dtype0*
_output_shapes
:
»
@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ï[q½*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel
»
@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ï[q=*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel

Jvgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBvgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel
¢
@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/subSub@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/max@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*
_output_shapes
: 
½
@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/mulMulJvgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/RandomUniform@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/sub*'
_output_shapes
:@*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel
¯
<vgg19_encoder/block2_conv1/kernel/Initializer/random_uniformAdd@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/mul@vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform/min*'
_output_shapes
:@*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel
¹
!vgg19_encoder/block2_conv1/kernel
VariableV2*
dtype0*'
_output_shapes
:@*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*
shape:@
û
(vgg19_encoder/block2_conv1/kernel/AssignAssign!vgg19_encoder/block2_conv1/kernel<vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*'
_output_shapes
:@
½
&vgg19_encoder/block2_conv1/kernel/readIdentity!vgg19_encoder/block2_conv1/kernel*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*'
_output_shapes
:@*
T0
´
1vgg19_encoder/block2_conv1/bias/Initializer/zerosConst*
valueB*    *2
_class(
&$loc:@vgg19_encoder/block2_conv1/bias*
dtype0*
_output_shapes	
:

vgg19_encoder/block2_conv1/bias
VariableV2*2
_class(
&$loc:@vgg19_encoder/block2_conv1/bias*
shape:*
dtype0*
_output_shapes	
:
Þ
&vgg19_encoder/block2_conv1/bias/AssignAssignvgg19_encoder/block2_conv1/bias1vgg19_encoder/block2_conv1/bias/Initializer/zeros*
T0*2
_class(
&$loc:@vgg19_encoder/block2_conv1/bias*
_output_shapes	
:
«
$vgg19_encoder/block2_conv1/bias/readIdentityvgg19_encoder/block2_conv1/bias*
T0*2
_class(
&$loc:@vgg19_encoder/block2_conv1/bias*
_output_shapes	
:
y
(vgg19_encoder/block2_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ò
!vgg19_encoder/block2_conv1/Conv2DConv2D!vgg19_encoder/block1_pool/MaxPool&vgg19_encoder/block2_conv1/kernel/read*
paddingSAME*)
_output_shapes
:*
T0*
strides

ª
"vgg19_encoder/block2_conv1/BiasAddBiasAdd!vgg19_encoder/block2_conv1/Conv2D$vgg19_encoder/block2_conv1/bias/read*
T0*)
_output_shapes
:

vgg19_encoder/block2_conv1/ReluRelu"vgg19_encoder/block2_conv1/BiasAdd*
T0*)
_output_shapes
:
Ñ
Bvgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel
»
@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ìQ½*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*
dtype0*
_output_shapes
: 
»
@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ìQ=*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel

Jvgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniformBvgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel
¢
@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/subSub@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/max@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel
¾
@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/mulMulJvgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/RandomUniform@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/sub*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*(
_output_shapes
:
°
<vgg19_encoder/block2_conv2/kernel/Initializer/random_uniformAdd@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/mul@vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*(
_output_shapes
:
»
!vgg19_encoder/block2_conv2/kernel
VariableV2*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*
shape:*
dtype0*(
_output_shapes
:
ü
(vgg19_encoder/block2_conv2/kernel/AssignAssign!vgg19_encoder/block2_conv2/kernel<vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*(
_output_shapes
:
¾
&vgg19_encoder/block2_conv2/kernel/readIdentity!vgg19_encoder/block2_conv2/kernel*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*(
_output_shapes
:
´
1vgg19_encoder/block2_conv2/bias/Initializer/zerosConst*
valueB*    *2
_class(
&$loc:@vgg19_encoder/block2_conv2/bias*
dtype0*
_output_shapes	
:

vgg19_encoder/block2_conv2/bias
VariableV2*2
_class(
&$loc:@vgg19_encoder/block2_conv2/bias*
shape:*
dtype0*
_output_shapes	
:
Þ
&vgg19_encoder/block2_conv2/bias/AssignAssignvgg19_encoder/block2_conv2/bias1vgg19_encoder/block2_conv2/bias/Initializer/zeros*2
_class(
&$loc:@vgg19_encoder/block2_conv2/bias*
_output_shapes	
:*
T0
«
$vgg19_encoder/block2_conv2/bias/readIdentityvgg19_encoder/block2_conv2/bias*
_output_shapes	
:*
T0*2
_class(
&$loc:@vgg19_encoder/block2_conv2/bias
y
(vgg19_encoder/block2_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ð
!vgg19_encoder/block2_conv2/Conv2DConv2Dvgg19_encoder/block2_conv1/Relu&vgg19_encoder/block2_conv2/kernel/read*)
_output_shapes
:*
T0*
strides
*
paddingSAME
ª
"vgg19_encoder/block2_conv2/BiasAddBiasAdd!vgg19_encoder/block2_conv2/Conv2D$vgg19_encoder/block2_conv2/bias/read*
T0*)
_output_shapes
:

vgg19_encoder/block2_conv2/ReluRelu"vgg19_encoder/block2_conv2/BiasAdd*
T0*)
_output_shapes
:
´
!vgg19_encoder/block2_pool/MaxPoolMaxPoolvgg19_encoder/block2_conv2/Relu*)
_output_shapes
:*
strides
*
ksize
*
paddingVALID
Ñ
Bvgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
dtype0*
_output_shapes
:
»
@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
dtype0*
_output_shapes
: 
»
@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *«ª*=*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
dtype0

Jvgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformBvgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/shape*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
dtype0*(
_output_shapes
:
¢
@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/subSub@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/max@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/min*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
_output_shapes
: 
¾
@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/mulMulJvgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/RandomUniform@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/sub*(
_output_shapes
:*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel
°
<vgg19_encoder/block3_conv1/kernel/Initializer/random_uniformAdd@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/mul@vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform/min*(
_output_shapes
:*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel
»
!vgg19_encoder/block3_conv1/kernel
VariableV2*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*
shape:*
dtype0*(
_output_shapes
:
ü
(vgg19_encoder/block3_conv1/kernel/AssignAssign!vgg19_encoder/block3_conv1/kernel<vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel*(
_output_shapes
:
¾
&vgg19_encoder/block3_conv1/kernel/readIdentity!vgg19_encoder/block3_conv1/kernel*(
_output_shapes
:*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel
´
1vgg19_encoder/block3_conv1/bias/Initializer/zerosConst*
valueB*    *2
_class(
&$loc:@vgg19_encoder/block3_conv1/bias*
dtype0*
_output_shapes	
:

vgg19_encoder/block3_conv1/bias
VariableV2*2
_class(
&$loc:@vgg19_encoder/block3_conv1/bias*
shape:*
dtype0*
_output_shapes	
:
Þ
&vgg19_encoder/block3_conv1/bias/AssignAssignvgg19_encoder/block3_conv1/bias1vgg19_encoder/block3_conv1/bias/Initializer/zeros*2
_class(
&$loc:@vgg19_encoder/block3_conv1/bias*
_output_shapes	
:*
T0
«
$vgg19_encoder/block3_conv1/bias/readIdentityvgg19_encoder/block3_conv1/bias*
T0*2
_class(
&$loc:@vgg19_encoder/block3_conv1/bias*
_output_shapes	
:
y
(vgg19_encoder/block3_conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ò
!vgg19_encoder/block3_conv1/Conv2DConv2D!vgg19_encoder/block2_pool/MaxPool&vgg19_encoder/block3_conv1/kernel/read*
paddingSAME*)
_output_shapes
:*
T0*
strides

ª
"vgg19_encoder/block3_conv1/BiasAddBiasAdd!vgg19_encoder/block3_conv1/Conv2D$vgg19_encoder/block3_conv1/bias/read*
T0*)
_output_shapes
:

vgg19_encoder/block3_conv1/ReluRelu"vgg19_encoder/block3_conv1/BiasAdd*
T0*)
_output_shapes
:
{
*vgg19_encoder_1/block1_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
¾
#vgg19_encoder_1/block1_conv1/Conv2DConv2DExpandDims_1&vgg19_encoder/block1_conv1/kernel/read*(
_output_shapes
:@*
T0*
strides
*
paddingSAME
­
$vgg19_encoder_1/block1_conv1/BiasAddBiasAdd#vgg19_encoder_1/block1_conv1/Conv2D$vgg19_encoder/block1_conv1/bias/read*(
_output_shapes
:@*
T0

!vgg19_encoder_1/block1_conv1/ReluRelu$vgg19_encoder_1/block1_conv1/BiasAdd*(
_output_shapes
:@*
T0
{
*vgg19_encoder_1/block1_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ó
#vgg19_encoder_1/block1_conv2/Conv2DConv2D!vgg19_encoder_1/block1_conv1/Relu&vgg19_encoder/block1_conv2/kernel/read*
strides
*
paddingSAME*(
_output_shapes
:@*
T0
­
$vgg19_encoder_1/block1_conv2/BiasAddBiasAdd#vgg19_encoder_1/block1_conv2/Conv2D$vgg19_encoder/block1_conv2/bias/read*
T0*(
_output_shapes
:@

!vgg19_encoder_1/block1_conv2/ReluRelu$vgg19_encoder_1/block1_conv2/BiasAdd*
T0*(
_output_shapes
:@
·
#vgg19_encoder_1/block1_pool/MaxPoolMaxPool!vgg19_encoder_1/block1_conv2/Relu*
strides
*
ksize
*
paddingVALID*(
_output_shapes
:@
{
*vgg19_encoder_1/block2_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ö
#vgg19_encoder_1/block2_conv1/Conv2DConv2D#vgg19_encoder_1/block1_pool/MaxPool&vgg19_encoder/block2_conv1/kernel/read*
T0*
strides
*
paddingSAME*)
_output_shapes
:
®
$vgg19_encoder_1/block2_conv1/BiasAddBiasAdd#vgg19_encoder_1/block2_conv1/Conv2D$vgg19_encoder/block2_conv1/bias/read*
T0*)
_output_shapes
:

!vgg19_encoder_1/block2_conv1/ReluRelu$vgg19_encoder_1/block2_conv1/BiasAdd*
T0*)
_output_shapes
:
{
*vgg19_encoder_1/block2_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ô
#vgg19_encoder_1/block2_conv2/Conv2DConv2D!vgg19_encoder_1/block2_conv1/Relu&vgg19_encoder/block2_conv2/kernel/read*
paddingSAME*)
_output_shapes
:*
T0*
strides

®
$vgg19_encoder_1/block2_conv2/BiasAddBiasAdd#vgg19_encoder_1/block2_conv2/Conv2D$vgg19_encoder/block2_conv2/bias/read*
T0*)
_output_shapes
:

!vgg19_encoder_1/block2_conv2/ReluRelu$vgg19_encoder_1/block2_conv2/BiasAdd*
T0*)
_output_shapes
:
¸
#vgg19_encoder_1/block2_pool/MaxPoolMaxPool!vgg19_encoder_1/block2_conv2/Relu*)
_output_shapes
:*
strides
*
ksize
*
paddingVALID
{
*vgg19_encoder_1/block3_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ö
#vgg19_encoder_1/block3_conv1/Conv2DConv2D#vgg19_encoder_1/block2_pool/MaxPool&vgg19_encoder/block3_conv1/kernel/read*
T0*
strides
*
paddingSAME*)
_output_shapes
:
®
$vgg19_encoder_1/block3_conv1/BiasAddBiasAdd#vgg19_encoder_1/block3_conv1/Conv2D$vgg19_encoder/block3_conv1/bias/read*)
_output_shapes
:*
T0

!vgg19_encoder_1/block3_conv1/ReluRelu$vgg19_encoder_1/block3_conv1/BiasAdd*
T0*)
_output_shapes
:
Ä
Sfast_style_transfer/transformation/transformation/base_image_transform/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
û
Jfast_style_transfer/transformation/transformation/base_image_transform/PadPadvgg19_encoder/block3_conv1/ReluSfast_style_transfer/transformation/transformation/base_image_transform/Pad/paddings*
T0*)
_output_shapes
:

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*
dtype0*
_output_shapes
:
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*
dtype0*
_output_shapes
: 
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *«ª*=*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel

 fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/shape*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*
dtype0*(
_output_shapes
:
ÿ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/mulMul fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/sub*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*(
_output_shapes
:

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform/min*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*(
_output_shapes
:
é
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel
VariableV2*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*
shape:*
dtype0*(
_output_shapes
:
×
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/AssignAssignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform*(
_output_shapes
:*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel
Â
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/readIdentitywfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*(
_output_shapes
:*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel
â
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Initializer/zerosConst*
valueB*    *
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias*
dtype0*
_output_shapes	
:
Ê
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias
¸
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/AssignAssignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Initializer/zeros*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias*
_output_shapes	
:
®
zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/readIdentityufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias*
_output_shapes	
:
Ï
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
¨
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/Conv2DConv2DJfast_style_transfer/transformation/transformation/base_image_transform/Pad|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/read*)
_output_shapes
:*
T0*
strides
*
paddingVALID
¬
xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/BiasAddBiasAddwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/Conv2Dzfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/read*)
_output_shapes
:*
T0
«
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/ReluReluxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/BiasAdd*
T0*)
_output_shapes
:
Æ
Ufast_style_transfer/transformation/transformation/base_image_transform/Pad_1/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Õ
Lfast_style_transfer/transformation/transformation/base_image_transform/Pad_1Padufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/ReluUfast_style_transfer/transformation/transformation/base_image_transform/Pad_1/paddings*
T0*)
_output_shapes
:

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*
dtype0*
_output_shapes
:
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ï[q½*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*
dtype0*
_output_shapes
: 
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ï[q=*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel

 fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel
ÿ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/mulMul fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/sub*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*'
_output_shapes
:@

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform/min*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*'
_output_shapes
:@
ç
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel
VariableV2*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*
shape:@*
dtype0*'
_output_shapes
:@
Ö
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/AssignAssignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*'
_output_shapes
:@
Á
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/readIdentitywfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel*'
_output_shapes
:@
à
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Initializer/zerosConst*
valueB@*    *
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias*
dtype0*
_output_shapes
:@
È
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias
·
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/AssignAssignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Initializer/zeros*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias*
_output_shapes
:@
­
zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/readIdentityufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias*
_output_shapes
:@
Ï
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
©
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/Conv2DConv2DLfast_style_transfer/transformation/transformation/base_image_transform/Pad_1|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/read*
T0*
strides
*
paddingVALID*(
_output_shapes
:@
«
xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/BiasAddBiasAddwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/Conv2Dzfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/read*
T0*(
_output_shapes
:@
ª
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/ReluReluxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/BiasAdd*(
_output_shapes
:@*
T0
Æ
Ufast_style_transfer/transformation/transformation/base_image_transform/Pad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ô
Lfast_style_transfer/transformation/transformation/base_image_transform/Pad_2Padufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/ReluUfast_style_transfer/transformation/transformation/base_image_transform/Pad_2/paddings*(
_output_shapes
:@*
T0

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @       *
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*
dtype0*
_output_shapes
:
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *«ªª½*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*
dtype0*
_output_shapes
: 
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ªª=*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*
dtype0*
_output_shapes
: 

 fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/shape*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*
dtype0*&
_output_shapes
:@ 
ÿ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/mulMul fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@ *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform/min*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*&
_output_shapes
:@ 
å
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel
VariableV2*
shape:@ *
dtype0*&
_output_shapes
:@ *
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel
Õ
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/AssignAssignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform*&
_output_shapes
:@ *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel
À
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/readIdentitywfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*&
_output_shapes
:@ 
à
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Initializer/zerosConst*
valueB *    *
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
dtype0*
_output_shapes
: 
È
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias
VariableV2*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
shape: *
dtype0*
_output_shapes
: 
·
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/AssignAssignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Initializer/zeros*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
_output_shapes
: 
­
zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/readIdentityufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
_output_shapes
: 
Ï
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
©
wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/Conv2DConv2DLfast_style_transfer/transformation/transformation/base_image_transform/Pad_2|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/read*
paddingVALID*(
_output_shapes
: *
T0*
strides

«
xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/BiasAddBiasAddwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/Conv2Dzfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/read*(
_output_shapes
: *
T0
ª
ufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/ReluReluxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/BiasAdd*
T0*(
_output_shapes
: 
¥
Lfast_style_transfer/transformation/transformation/base_image_transform/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
¤
Zfast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¦
\fast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¦
\fast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Tfast_style_transfer/transformation/transformation/base_image_transform/strided_sliceStridedSliceLfast_style_transfer/transformation/transformation/base_image_transform/ShapeZfast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stack\fast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stack_1\fast_style_transfer/transformation/transformation/base_image_transform/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
¡
Vfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shape/1Const*
_output_shapes
: *
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0

Vfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shape/2Const*
dtype0*
_output_shapes
: *
value	B : 

Tfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shapePackTfast_style_transfer/transformation/transformation/base_image_transform/strided_sliceVfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shape/1Vfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shape/2*
T0*
N*
_output_shapes
:
Õ
Nfast_style_transfer/transformation/transformation/base_image_transform/ReshapeReshapeufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/ReluTfast_style_transfer/transformation/transformation/base_image_transform/Reshape/shape*$
_output_shapes
: *
T0
¯
efast_style_transfer/transformation/transformation/base_image_transform/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ð
Sfast_style_transfer/transformation/transformation/base_image_transform/moments/meanMeanNfast_style_transfer/transformation/transformation/base_image_transform/Reshapeefast_style_transfer/transformation/transformation/base_image_transform/moments/mean/reduction_indices*
	keep_dims(*
T0*"
_output_shapes
: 
í
[fast_style_transfer/transformation/transformation/base_image_transform/moments/StopGradientStopGradientSfast_style_transfer/transformation/transformation/base_image_transform/moments/mean*"
_output_shapes
: *
T0
Ñ
`fast_style_transfer/transformation/transformation/base_image_transform/moments/SquaredDifferenceSquaredDifferenceNfast_style_transfer/transformation/transformation/base_image_transform/Reshape[fast_style_transfer/transformation/transformation/base_image_transform/moments/StopGradient*$
_output_shapes
: *
T0
³
ifast_style_transfer/transformation/transformation/base_image_transform/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ê
Wfast_style_transfer/transformation/transformation/base_image_transform/moments/varianceMean`fast_style_transfer/transformation/transformation/base_image_transform/moments/SquaredDifferenceifast_style_transfer/transformation/transformation/base_image_transform/moments/variance/reduction_indices*
T0*"
_output_shapes
: *
	keep_dims(

Vfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
¹
Tfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/addAddWfast_style_transfer/transformation/transformation/base_image_transform/moments/varianceVfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/add/y*
T0*"
_output_shapes
: 
â
Vfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/RsqrtRsqrtTfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/add*"
_output_shapes
: *
T0
²
Tfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/mulMulNfast_style_transfer/transformation/transformation/base_image_transform/ReshapeVfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/Rsqrt*
T0*$
_output_shapes
: 
Ý
Tfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/NegNegSfast_style_transfer/transformation/transformation/base_image_transform/moments/mean*"
_output_shapes
: *
T0
¸
Vfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/mul_1MulTfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/NegVfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/Rsqrt*"
_output_shapes
: *
T0
º
Vfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/add_1AddTfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/mulVfast_style_transfer/transformation/transformation/base_image_transform/batchnorm/mul_1*$
_output_shapes
: *
T0
¥
Jfast_style_transfer/transformation/transformation/base_image_transform/subSubNfast_style_transfer/transformation/transformation/base_image_transform/ReshapeSfast_style_transfer/transformation/transformation/base_image_transform/moments/mean*
T0*$
_output_shapes
: 

Pfast_style_transfer/transformation/transformation/base_image_transform/truediv/yConst*
valueB
 *   C*
dtype0*
_output_shapes
: 
¦
Nfast_style_transfer/transformation/transformation/base_image_transform/truedivRealDivJfast_style_transfer/transformation/transformation/base_image_transform/subPfast_style_transfer/transformation/transformation/base_image_transform/truediv/y*
T0*$
_output_shapes
: 
£
Nfast_style_transfer/transformation/transformation/base_image_transform/Shape_1Const*!
valueB"    @      *
dtype0*
_output_shapes
:
Ø
Nfast_style_transfer/transformation/transformation/base_image_transform/unstackUnpackNfast_style_transfer/transformation/transformation/base_image_transform/Shape_1*
_output_shapes
: : : *
T0*	
num
Ø
Nfast_style_transfer/transformation/transformation/base_image_transform/ToFloatCastPfast_style_transfer/transformation/transformation/base_image_transform/unstack:1*

DstT0*
_output_shapes
: *

SrcT0

Lfast_style_transfer/transformation/transformation/base_image_transform/add/yConst*
valueB
 *æ$*
dtype0*
_output_shapes
: 

Jfast_style_transfer/transformation/transformation/base_image_transform/addAddNfast_style_transfer/transformation/transformation/base_image_transform/ToFloatLfast_style_transfer/transformation/transformation/base_image_transform/add/y*
T0*
_output_shapes
: 
¶
Mfast_style_transfer/transformation/transformation/base_image_transform/MatMulBatchMatMulNfast_style_transfer/transformation/transformation/base_image_transform/truedivNfast_style_transfer/transformation/transformation/base_image_transform/truediv*
adj_x(*
T0*"
_output_shapes
:  
£
Pfast_style_transfer/transformation/transformation/base_image_transform/truediv_1RealDivMfast_style_transfer/transformation/transformation/base_image_transform/MatMulJfast_style_transfer/transformation/transformation/base_image_transform/add*
T0*"
_output_shapes
:  
ú
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/shapeConst*
valueB"       *
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
dtype0*
_output_shapes
:
ì
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/minConst*
valueB
 *A×½*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
dtype0*
_output_shapes
: 
ì
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/maxConst*
valueB
 *A×=*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
dtype0*
_output_shapes
: 

¡fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/min*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
_output_shapes
: 

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/mulMul¡fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/sub*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
_output_shapes
:	 

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform/min*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
_output_shapes
:	 
Ù
xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel
VariableV2*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
shape:	 *
dtype0*
_output_shapes
:	 
Ò
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/AssignAssignxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform*
_output_shapes
:	 *
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel
¼
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/readIdentityxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel*
_output_shapes
:	 
ä
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Initializer/zerosConst*
valueB*    *
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias*
dtype0*
_output_shapes	
:
Ì
vfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias
VariableV2*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias*
shape:*
dtype0*
_output_shapes	
:
¼
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/AssignAssignvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Initializer/zeros*
T0*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias*
_output_shapes	
:
±
{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/readIdentityvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias*
_output_shapes	
:*
T0
à
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose	TransposePfast_style_transfer/transformation/transformation/base_image_transform/truediv_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose/perm*"
_output_shapes
:  *
T0
Û
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape/shapeConst*
valueB"        *
dtype0*
_output_shapes
:
Ì
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transposefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape/shape*
T0*
_output_shapes

:  
Þ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Í
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose_1	Transpose}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/readfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose_1/perm*
_output_shapes
:	 *
T0
Ý
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ó
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/transpose_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape_1/shape*
T0*
_output_shapes
:	 
Å
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/Reshape_1*
T0*
_output_shapes
:	 
×
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/shapeConst*!
valueB"          *
dtype0*
_output_shapes
:
½
{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/TensordotReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/MatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot/shape*
T0*#
_output_shapes
: 
¬
yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/BiasAddBiasAdd{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Tensordot{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/read*
T0*#
_output_shapes
: 
§
vfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/ReluReluyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/BiasAdd*#
_output_shapes
: *
T0
µ
qfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/Initializer/onesConst*
valueB*  ?*s
_classi
geloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma*
dtype0*
_output_shapes	
:

`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*s
_classi
geloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma
á
gfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/AssignAssign`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gammaqfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/Initializer/ones*
T0*s
_classi
geloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma*
_output_shapes	
:
î
efast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/readIdentity`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma*
T0*s
_classi
geloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma*
_output_shapes	
:
´
qfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/Initializer/zerosConst*
valueB*    *r
_classh
fdloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta*
dtype0*
_output_shapes	
:

_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta
VariableV2*r
_classh
fdloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta*
shape:*
dtype0*
_output_shapes	
:
Þ
ffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/AssignAssign_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/betaqfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/Initializer/zeros*
_output_shapes	
:*
T0*r
_classh
fdloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta
ë
dfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/readIdentity_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta*
_output_shapes	
:*
T0*r
_classh
fdloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta
Â
xfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/Initializer/zerosConst*
valueB*    *y
_classo
mkloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean*
dtype0*
_output_shapes	
:
«
ffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean
VariableV2*y
_classo
mkloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean*
shape:*
dtype0*
_output_shapes	
:
ú
mfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/AssignAssignffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_meanxfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:*
T0*y
_classo
mkloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean

kfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/readIdentityffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean*
_output_shapes	
:*
T0*y
_classo
mkloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean
É
{fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/Initializer/onesConst*
valueB*  ?*}
_classs
qoloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance*
dtype0*
_output_shapes	
:
³
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*}
_classs
qoloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance

qfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/AssignAssignjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance{fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/Initializer/ones*
T0*}
_classs
qoloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance*
_output_shapes	
:

ofast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/readIdentityjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance*
T0*}
_classs
qoloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance*
_output_shapes	
:
¯
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ò
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/addAddofast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/readjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/RsqrtRsqrthfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
è
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mulMuljfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/Rsqrtefast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/read*
_output_shapes	
:*
T0

jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul_1Mulvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/Reluhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul*
T0*#
_output_shapes
: 
î
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul_2Mulkfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/readhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
ç
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/subSubdfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/readjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
õ
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/add_1Addjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/mul_1hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/sub*
T0*#
_output_shapes
: 
ú
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
dtype0*
_output_shapes
:
ì
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/minConst*
valueB
 *øKÆ½*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
dtype0*
_output_shapes
: 
ì
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/maxConst*
valueB
 *øKÆ=*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
dtype0*
_output_shapes
: 

¡fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/min*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
_output_shapes
: 

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/mulMul¡fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/sub*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel* 
_output_shapes
:


fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform/min*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel* 
_output_shapes
:
*
T0
Û
xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel
VariableV2*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
shape:
*
dtype0* 
_output_shapes
:

Ó
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/AssignAssignxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform* 
_output_shapes
:
*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel
½
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/readIdentityxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel* 
_output_shapes
:

ä
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Initializer/zerosConst*
valueB*    *
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
dtype0*
_output_shapes	
:
Ì
vfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias
VariableV2*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
shape:*
dtype0*
_output_shapes	
:
¼
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/AssignAssignvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Initializer/zeros*
T0*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
_output_shapes	
:
±
{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/readIdentityvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
_output_shapes	
:*
T0
à
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
º
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose	Transposejfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/batchnorm/add_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose/perm*
T0*#
_output_shapes
: 
Û
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Í
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transposefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape/shape*
T0*
_output_shapes
:	 
Þ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
Î
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose_1	Transpose}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/readfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose_1/perm* 
_output_shapes
:
*
T0
Ý
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Ô
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/transpose_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape_1/shape*
T0* 
_output_shapes
:

Å
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/Reshape_1*
T0*
_output_shapes
:	 
×
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/shapeConst*
_output_shapes
:*!
valueB"          *
dtype0
½
{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/TensordotReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/MatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot/shape*#
_output_shapes
: *
T0
¬
yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/BiasAddBiasAdd{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Tensordot{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/read*#
_output_shapes
: *
T0
§
vfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/ReluReluyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/BiasAdd*
T0*#
_output_shapes
: 
¹
sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/Initializer/onesConst*
valueB*  ?*u
_classk
igloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
dtype0*
_output_shapes	
:
£
bfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma
VariableV2*u
_classk
igloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
shape:*
dtype0*
_output_shapes	
:
é
ifast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/AssignAssignbfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gammasfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/Initializer/ones*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
_output_shapes	
:
ô
gfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/readIdentitybfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
_output_shapes	
:
¸
sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/Initializer/zerosConst*
valueB*    *t
_classj
hfloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta*
dtype0*
_output_shapes	
:
¡
afast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta
VariableV2*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta*
shape:*
dtype0*
_output_shapes	
:
æ
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/AssignAssignafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/betasfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/Initializer/zeros*
T0*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta*
_output_shapes	
:
ñ
ffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/readIdentityafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta*
_output_shapes	
:*
T0*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta
Æ
zfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *{
_classq
omloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean*
dtype0
¯
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*{
_classq
omloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean

ofast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/AssignAssignhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_meanzfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/Initializer/zeros*
T0*{
_classq
omloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:

mfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/readIdentityhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean*
T0*{
_classq
omloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:
Í
}fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/Initializer/onesConst*
valueB*  ?*
_classu
sqloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:
·
lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance
VariableV2*
_classu
sqloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance*
shape:*
dtype0*
_output_shapes	
:

sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/AssignAssignlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance}fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/Initializer/ones*
T0*
_classu
sqloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance*
_output_shapes	
:

qfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/readIdentitylfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance*
_output_shapes	
:*
T0*
_classu
sqloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance
±
lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ø
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/addAddqfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/readlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes	
:

lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/RsqrtRsqrtjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/add*
T0*
_output_shapes	
:
î
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mulMullfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/Rsqrtgfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/read*
_output_shapes	
:*
T0

lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul_1Mulvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/Relujfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul*
T0*#
_output_shapes
: 
ô
lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul_2Mulmfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/readjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul*
T0*
_output_shapes	
:
í
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/subSubffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/readlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul_2*
_output_shapes	
:*
T0
û
lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/add_1Addlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/mul_1jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/sub*
T0*#
_output_shapes
: 

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/shapeConst*
valueB"       *
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
dtype0*
_output_shapes
:
ô
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/minConst*
valueB
 *øKF¾*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
dtype0*
_output_shapes
: 
ô
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/maxConst*
valueB
 *øKF>*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
dtype0*
_output_shapes
: 
 
¤fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel
¤
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/mulMul¤fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
_output_shapes
:	 

fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
_output_shapes
:	 
á
{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel
VariableV2*
shape:	 *
dtype0*
_output_shapes
:	 *
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel
á
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/AssignAssign{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernelfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
_output_shapes
:	 
È
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/readIdentity{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel*
_output_shapes
:	 *
T0
ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Initializer/zerosConst*
valueB *    *
_class
~loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
dtype0*
_output_shapes
: 
Ò
yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias
VariableV2*
_class
~loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
shape: *
dtype0*
_output_shapes
: 
Ê
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/AssignAssignyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/biasfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Initializer/zeros*
_class
~loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
_output_shapes
: *
T0
»
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/readIdentityyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
_output_shapes
: 
ã
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Â
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose	Transposelfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/batchnorm/add_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose/perm*
T0*#
_output_shapes
: 
Þ
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ö
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transposefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape/shape*
_output_shapes
:	 *
T0
á
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
×
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose_1	Transposefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/readfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose_1/perm*
T0*
_output_shapes
:	 
à
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ü
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/transpose_1fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape_1/shape*
_output_shapes
:	 *
T0
Í
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/Reshape_1*
T0*
_output_shapes

:  
Ú
fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/shapeConst*!
valueB"           *
dtype0*
_output_shapes
:
Å
~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/TensordotReshapefast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/MatMulfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot/shape*
T0*"
_output_shapes
:  
´
|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/BiasAddBiasAdd~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tensordot~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/read*
T0*"
_output_shapes
:  
¬
yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/TanhTanh|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/BiasAdd*
T0*"
_output_shapes
:  
Å
Tfast_style_transfer/transformation/transformation/style_image_transform/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
ÿ
Kfast_style_transfer/transformation/transformation/style_image_transform/PadPad!vgg19_encoder_1/block3_conv1/ReluTfast_style_transfer/transformation/transformation/style_image_transform/Pad/paddings*)
_output_shapes
:*
T0

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
dtype0*
_output_shapes
:
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
dtype0*
_output_shapes
: 
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ª*=*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
dtype0*
_output_shapes
: 
¢
¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/shape*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
dtype0*(
_output_shapes
:

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel
¤
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/mulMul¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/sub*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*(
_output_shapes
:

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform/min*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*(
_output_shapes
:
î
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel
VariableV2*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
shape:*
dtype0*(
_output_shapes
:
á
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/AssignAssignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*(
_output_shapes
:
É
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/readIdentityyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*(
_output_shapes
:
ç
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias
Ï
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias
VariableV2*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias*
shape:*
dtype0*
_output_shapes	
:
Á
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/AssignAssignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Initializer/zeros*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias*
_output_shapes	
:
µ
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/readIdentitywfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias*
_output_shapes	
:
Ò
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
­
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/Conv2DConv2DKfast_style_transfer/transformation/transformation/style_image_transform/Pad~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/read*)
_output_shapes
:*
T0*
strides
*
paddingVALID
²
zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/BiasAddBiasAddyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/Conv2D|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/read*)
_output_shapes
:*
T0
¯
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/ReluReluzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/BiasAdd*)
_output_shapes
:*
T0
Ç
Vfast_style_transfer/transformation/transformation/style_image_transform/Pad_1/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ù
Mfast_style_transfer/transformation/transformation/style_image_transform/Pad_1Padwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/ReluVfast_style_transfer/transformation/transformation/style_image_transform/Pad_1/paddings*)
_output_shapes
:*
T0

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
dtype0*
_output_shapes
:
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ï[q½*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
dtype0*
_output_shapes
: 
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ï[q=*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
dtype0*
_output_shapes
: 
¡
¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/shape*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
dtype0*'
_output_shapes
:@

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/min*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
_output_shapes
: 
£
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/mulMul¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/sub*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*'
_output_shapes
:@

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform/min*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*'
_output_shapes
:@
ì
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel
VariableV2*
shape:@*
dtype0*'
_output_shapes
:@*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel
à
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/AssignAssignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*'
_output_shapes
:@
È
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/readIdentityyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*'
_output_shapes
:@
å
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Initializer/zerosConst*
valueB@*    *
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias*
dtype0*
_output_shapes
:@
Í
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias
À
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/AssignAssignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Initializer/zeros*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias*
_output_shapes
:@
´
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/readIdentitywfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias*
_output_shapes
:@*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias
Ò
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
®
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/Conv2DConv2DMfast_style_transfer/transformation/transformation/style_image_transform/Pad_1~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/read*
paddingVALID*(
_output_shapes
:@*
T0*
strides

±
zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/BiasAddBiasAddyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/Conv2D|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/read*
T0*(
_output_shapes
:@
®
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/ReluReluzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/BiasAdd*(
_output_shapes
:@*
T0
Ç
Vfast_style_transfer/transformation/transformation/style_image_transform/Pad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ø
Mfast_style_transfer/transformation/transformation/style_image_transform/Pad_2Padwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/ReluVfast_style_transfer/transformation/transformation/style_image_transform/Pad_2/paddings*(
_output_shapes
:@*
T0

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @       *
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
dtype0*
_output_shapes
:
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *«ªª½*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
dtype0*
_output_shapes
: 
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ªª=*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
dtype0*
_output_shapes
: 
 
¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/shape*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
dtype0*&
_output_shapes
:@ 

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/min*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
_output_shapes
: 
¢
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/mulMul¢fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/sub*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*&
_output_shapes
:@ 

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform/min*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*&
_output_shapes
:@ 
ê
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel
VariableV2*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*
shape:@ *
dtype0*&
_output_shapes
:@ 
ß
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/AssignAssignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*&
_output_shapes
:@ *
T0
Ç
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/readIdentityyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel*&
_output_shapes
:@ *
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel
å
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Initializer/zerosConst*
valueB *    *
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias*
dtype0*
_output_shapes
: 
Í
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias
VariableV2*
dtype0*
_output_shapes
: *
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias*
shape: 
À
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/AssignAssignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Initializer/zeros*
_output_shapes
: *
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias
´
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/readIdentitywfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias*
_output_shapes
: 
Ò
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
®
yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/Conv2DConv2DMfast_style_transfer/transformation/transformation/style_image_transform/Pad_2~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/read*(
_output_shapes
: *
T0*
strides
*
paddingVALID
±
zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/BiasAddBiasAddyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/Conv2D|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/read*
T0*(
_output_shapes
: 
®
wfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/ReluReluzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/BiasAdd*
T0*(
_output_shapes
: 
¦
Mfast_style_transfer/transformation/transformation/style_image_transform/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
¥
[fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
§
]fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
§
]fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Ufast_style_transfer/transformation/transformation/style_image_transform/strided_sliceStridedSliceMfast_style_transfer/transformation/transformation/style_image_transform/Shape[fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stack]fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stack_1]fast_style_transfer/transformation/transformation/style_image_transform/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
¢
Wfast_style_transfer/transformation/transformation/style_image_transform/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 

Wfast_style_transfer/transformation/transformation/style_image_transform/Reshape/shape/2Const*
_output_shapes
: *
value	B : *
dtype0

Ufast_style_transfer/transformation/transformation/style_image_transform/Reshape/shapePackUfast_style_transfer/transformation/transformation/style_image_transform/strided_sliceWfast_style_transfer/transformation/transformation/style_image_transform/Reshape/shape/1Wfast_style_transfer/transformation/transformation/style_image_transform/Reshape/shape/2*
T0*
N*
_output_shapes
:
Ù
Ofast_style_transfer/transformation/transformation/style_image_transform/ReshapeReshapewfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/ReluUfast_style_transfer/transformation/transformation/style_image_transform/Reshape/shape*
T0*$
_output_shapes
: 
°
ffast_style_transfer/transformation/transformation/style_image_transform/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
Ó
Tfast_style_transfer/transformation/transformation/style_image_transform/moments/meanMeanOfast_style_transfer/transformation/transformation/style_image_transform/Reshapeffast_style_transfer/transformation/transformation/style_image_transform/moments/mean/reduction_indices*
	keep_dims(*
T0*"
_output_shapes
: 
ï
\fast_style_transfer/transformation/transformation/style_image_transform/moments/StopGradientStopGradientTfast_style_transfer/transformation/transformation/style_image_transform/moments/mean*
T0*"
_output_shapes
: 
Ô
afast_style_transfer/transformation/transformation/style_image_transform/moments/SquaredDifferenceSquaredDifferenceOfast_style_transfer/transformation/transformation/style_image_transform/Reshape\fast_style_transfer/transformation/transformation/style_image_transform/moments/StopGradient*$
_output_shapes
: *
T0
´
jfast_style_transfer/transformation/transformation/style_image_transform/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
í
Xfast_style_transfer/transformation/transformation/style_image_transform/moments/varianceMeanafast_style_transfer/transformation/transformation/style_image_transform/moments/SquaredDifferencejfast_style_transfer/transformation/transformation/style_image_transform/moments/variance/reduction_indices*
	keep_dims(*
T0*"
_output_shapes
: 

Wfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
¼
Ufast_style_transfer/transformation/transformation/style_image_transform/batchnorm/addAddXfast_style_transfer/transformation/transformation/style_image_transform/moments/varianceWfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/add/y*
T0*"
_output_shapes
: 
ä
Wfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/RsqrtRsqrtUfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/add*
T0*"
_output_shapes
: 
µ
Ufast_style_transfer/transformation/transformation/style_image_transform/batchnorm/mulMulOfast_style_transfer/transformation/transformation/style_image_transform/ReshapeWfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/Rsqrt*
T0*$
_output_shapes
: 
ß
Ufast_style_transfer/transformation/transformation/style_image_transform/batchnorm/NegNegTfast_style_transfer/transformation/transformation/style_image_transform/moments/mean*"
_output_shapes
: *
T0
»
Wfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/mul_1MulUfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/NegWfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/Rsqrt*
T0*"
_output_shapes
: 
½
Wfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/add_1AddUfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/mulWfast_style_transfer/transformation/transformation/style_image_transform/batchnorm/mul_1*
T0*$
_output_shapes
: 
¨
Kfast_style_transfer/transformation/transformation/style_image_transform/subSubOfast_style_transfer/transformation/transformation/style_image_transform/ReshapeTfast_style_transfer/transformation/transformation/style_image_transform/moments/mean*
T0*$
_output_shapes
: 

Qfast_style_transfer/transformation/transformation/style_image_transform/truediv/yConst*
valueB
 *   C*
dtype0*
_output_shapes
: 
©
Ofast_style_transfer/transformation/transformation/style_image_transform/truedivRealDivKfast_style_transfer/transformation/transformation/style_image_transform/subQfast_style_transfer/transformation/transformation/style_image_transform/truediv/y*
T0*$
_output_shapes
: 
¤
Ofast_style_transfer/transformation/transformation/style_image_transform/Shape_1Const*!
valueB"    @      *
dtype0*
_output_shapes
:
Ú
Ofast_style_transfer/transformation/transformation/style_image_transform/unstackUnpackOfast_style_transfer/transformation/transformation/style_image_transform/Shape_1*
_output_shapes
: : : *
T0*	
num
Ú
Ofast_style_transfer/transformation/transformation/style_image_transform/ToFloatCastQfast_style_transfer/transformation/transformation/style_image_transform/unstack:1*

SrcT0*

DstT0*
_output_shapes
: 

Mfast_style_transfer/transformation/transformation/style_image_transform/add/yConst*
valueB
 *æ$*
dtype0*
_output_shapes
: 

Kfast_style_transfer/transformation/transformation/style_image_transform/addAddOfast_style_transfer/transformation/transformation/style_image_transform/ToFloatMfast_style_transfer/transformation/transformation/style_image_transform/add/y*
T0*
_output_shapes
: 
¹
Nfast_style_transfer/transformation/transformation/style_image_transform/MatMulBatchMatMulOfast_style_transfer/transformation/transformation/style_image_transform/truedivOfast_style_transfer/transformation/transformation/style_image_transform/truediv*
adj_x(*
T0*"
_output_shapes
:  
¦
Qfast_style_transfer/transformation/transformation/style_image_transform/truediv_1RealDivNfast_style_transfer/transformation/transformation/style_image_transform/MatMulKfast_style_transfer/transformation/transformation/style_image_transform/add*"
_output_shapes
:  *
T0
ÿ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"       *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel
ñ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *A×½*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel
ñ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/maxConst*
valueB
 *A×=*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
dtype0*
_output_shapes
: 

£fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
: 

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/mulMul£fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
:	 

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform/min*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
:	 *
T0
Þ
zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel
VariableV2*
dtype0*
_output_shapes
:	 *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
shape:	 
Ü
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/AssignAssignzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
:	 
Ã
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/readIdentityzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
:	 
é
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias
Ñ
xfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias
VariableV2*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias*
shape:*
dtype0*
_output_shapes	
:
Å
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/AssignAssignxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Initializer/zeros*
_output_shapes	
:*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias
¸
}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/readIdentityxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias*
_output_shapes	
:
â
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0
¤
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose	TransposeQfast_style_transfer/transformation/transformation/style_image_transform/truediv_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose/perm*"
_output_shapes
:  *
T0
Ý
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape/shapeConst*
valueB"        *
dtype0*
_output_shapes
:
Ò
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape/shape*
T0*
_output_shapes

:  
à
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose_1/permConst*
_output_shapes
:*
valueB"       *
dtype0
Ó
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose_1	Transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/readfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose_1/perm*
_output_shapes
:	 *
T0
ß
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ù
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/transpose_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape_1/shape*
T0*
_output_shapes
:	 
Ë
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/Reshape_1*
_output_shapes
:	 *
T0
Ù
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/shapeConst*
dtype0*
_output_shapes
:*!
valueB"          
Ã
}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/TensordotReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/MatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot/shape*#
_output_shapes
: *
T0
²
{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/BiasAddBiasAdd}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Tensordot}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/read*
T0*#
_output_shapes
: 
«
xfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/ReluRelu{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/BiasAdd*
T0*#
_output_shapes
: 
·
rfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/Initializer/onesConst*
valueB*  ?*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
dtype0*
_output_shapes	
:
¡
afast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
shape:
å
hfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/AssignAssignafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gammarfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/Initializer/ones*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
_output_shapes	
:*
T0
ñ
ffast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/readIdentityafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
T0*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
_output_shapes	
:
¶
rfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *s
_classi
geloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta

`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta
VariableV2*s
_classi
geloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta*
shape:*
dtype0*
_output_shapes	
:
â
gfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/AssignAssign`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/betarfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/Initializer/zeros*
T0*s
_classi
geloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta*
_output_shapes	
:
î
efast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/readIdentity`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta*s
_classi
geloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta*
_output_shapes	
:*
T0
Ä
yfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/Initializer/zerosConst*
valueB*    *z
_classp
nlloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean*
dtype0*
_output_shapes	
:
­
gfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*z
_classp
nlloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean
þ
nfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/AssignAssigngfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_meanyfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/Initializer/zeros*
T0*z
_classp
nlloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean*
_output_shapes	
:

lfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/readIdentitygfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean*
_output_shapes	
:*
T0*z
_classp
nlloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean
Ë
|fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/Initializer/onesConst*
valueB*  ?*~
_classt
rploc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance*
dtype0*
_output_shapes	
:
µ
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*~
_classt
rploc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance

rfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/AssignAssignkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance|fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/Initializer/ones*
T0*~
_classt
rploc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance*
_output_shapes	
:

pfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/readIdentitykfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance*
T0*~
_classt
rploc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance*
_output_shapes	
:
°
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
õ
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/addAddpfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/readkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/RsqrtRsqrtifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0
ë
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mulMulkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/Rsqrtffast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/read*
T0*
_output_shapes	
:

kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul_1Mulxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/Reluifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul*#
_output_shapes
: *
T0
ñ
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul_2Mullfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/readifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
ê
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/subSubefast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/readkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ø
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/add_1Addkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/mul_1ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/sub*
T0*#
_output_shapes
: 
ÿ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel*
dtype0*
_output_shapes
:
ñ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *øKÆ½*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
ñ
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/maxConst*
valueB
 *øKÆ=*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel*
dtype0*
_output_shapes
: 

£fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel*
dtype0* 
_output_shapes
:


fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
 
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/mulMul£fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel* 
_output_shapes
:


fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
à
zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
Ý
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/AssignAssignzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform* 
_output_shapes
:
*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
Ä
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/readIdentityzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel* 
_output_shapes
:
*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel
é
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Initializer/zerosConst*
valueB*    *
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
dtype0*
_output_shapes	
:
Ñ
xfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias
VariableV2*
dtype0*
_output_shapes	
:*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
shape:
Å
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/AssignAssignxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Initializer/zeros*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
_output_shapes	
:
¸
}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/readIdentityxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
_output_shapes	
:*
T0
â
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
¿
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose	Transposekfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/batchnorm/add_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose/perm*#
_output_shapes
: *
T0
Ý
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ó
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape/shape*
T0*
_output_shapes
:	 
à
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
Ô
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose_1	Transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/readfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose_1/perm*
T0* 
_output_shapes
:

ß
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
Ú
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/transpose_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape_1/shape*
T0* 
_output_shapes
:

Ë
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/Reshape_1*
T0*
_output_shapes
:	 
Ù
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/shapeConst*!
valueB"          *
dtype0*
_output_shapes
:
Ã
}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/TensordotReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/MatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot/shape*#
_output_shapes
: *
T0
²
{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/BiasAddBiasAdd}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Tensordot}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/read*
T0*#
_output_shapes
: 
«
xfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/ReluRelu{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/BiasAdd*#
_output_shapes
: *
T0
»
tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/Initializer/onesConst*
valueB*  ?*v
_classl
jhloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma*
dtype0*
_output_shapes	
:
¥
cfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma
VariableV2*v
_classl
jhloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma*
shape:*
dtype0*
_output_shapes	
:
í
jfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/AssignAssigncfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gammatfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/Initializer/ones*
_output_shapes	
:*
T0*v
_classl
jhloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma
÷
hfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/readIdentitycfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma*v
_classl
jhloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma*
_output_shapes	
:*
T0
º
tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/Initializer/zerosConst*
valueB*    *u
_classk
igloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta*
dtype0*
_output_shapes	
:
£
bfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta
VariableV2*
dtype0*
_output_shapes	
:*u
_classk
igloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta*
shape:
ê
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/AssignAssignbfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/betatfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/Initializer/zeros*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta*
_output_shapes	
:
ô
gfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/readIdentitybfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta*
_output_shapes	
:*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta
È
{fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB*    *|
_classr
pnloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
dtype0*
_output_shapes	
:
±
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*|
_classr
pnloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
shape:

pfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/AssignAssignifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean{fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/Initializer/zeros*
T0*|
_classr
pnloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:

nfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/readIdentityifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
T0*|
_classr
pnloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:
Ð
~fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/Initializer/onesConst*
valueB*  ?*
_classv
trloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance*
dtype0*
_output_shapes	
:
º
mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance
VariableV2*
_classv
trloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance*
shape:*
dtype0*
_output_shapes	
:

tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/AssignAssignmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance~fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/Initializer/ones*
T0*
_classv
trloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance*
_output_shapes	
:

rfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/readIdentitymfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance*
T0*
_classv
trloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance*
_output_shapes	
:
²
mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
û
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/addAddrfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/readmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/add/y*
T0*
_output_shapes	
:

mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/RsqrtRsqrtkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/add*
T0*
_output_shapes	
:
ñ
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mulMulmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/Rsqrthfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/read*
T0*
_output_shapes	
:

mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul_1Mulxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/Relukfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul*#
_output_shapes
: *
T0
÷
mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul_2Mulnfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/readkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul*
_output_shapes	
:*
T0
ð
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/subSubgfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/readmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul_2*
T0*
_output_shapes	
:
þ
mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/add_1Addmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/mul_1kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/sub*#
_output_shapes
: *
T0

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"       *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel
ø
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/minConst*
valueB
 *øKF¾*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
dtype0*
_output_shapes
: 
ø
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/maxConst*
valueB
 *øKF>*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
dtype0*
_output_shapes
: 
¦
¦fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/RandomUniformRandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	 *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/subSubfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/maxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
_output_shapes
: 
¬
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/mulMul¦fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/RandomUniformfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/sub*
_output_shapes
:	 *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel

fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniformAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/mulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
_output_shapes
:	 
å
}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel
VariableV2*
dtype0*
_output_shapes
:	 *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
shape:	 
é
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/AssignAssign}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernelfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
_output_shapes
:	 
Î
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/readIdentity}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
_output_shapes
:	 
ï
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Initializer/zerosConst*
valueB *    *
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
dtype0*
_output_shapes
: 
×
{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias
VariableV2*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
shape: *
dtype0*
_output_shapes
: 
Ó
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/AssignAssign{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/biasfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Initializer/zeros*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
_output_shapes
: 
Ã
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/readIdentity{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
_output_shapes
: 
å
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:
Ç
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose	Transposemfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/batchnorm/add_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose/perm*#
_output_shapes
: *
T0
à
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Ü
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/ReshapeReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape/shape*
T0*
_output_shapes
:	 
ã
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
Ý
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose_1	Transposefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/readfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose_1/perm*
_output_shapes
:	 *
T0
â
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
â
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape_1Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/transpose_1fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape_1/shape*
_output_shapes
:	 *
T0
Ó
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/MatMulMatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/Reshape_1*
T0*
_output_shapes

:  
Ü
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/shapeConst*!
valueB"           *
dtype0*
_output_shapes
:
Ì
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/TensordotReshapefast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/MatMulfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordot/shape*
T0*"
_output_shapes
:  
¼
~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/BiasAddBiasAddfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tensordotfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/read*
T0*"
_output_shapes
:  
°
{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/TanhTanh~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/BiasAdd*
T0*"
_output_shapes
:  

8fast_style_transfer/transformation/einsum/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          
¢
3fast_style_transfer/transformation/einsum/transpose	Transposeyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/Tanh8fast_style_transfer/transformation/einsum/transpose/perm*
T0*"
_output_shapes
:  

:fast_style_transfer/transformation/einsum/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:
¨
5fast_style_transfer/transformation/einsum/transpose_1	Transpose{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/Tanh:fast_style_transfer/transformation/einsum/transpose_1/perm*
T0*"
_output_shapes
:  
Ø
0fast_style_transfer/transformation/einsum/MatMulBatchMatMul3fast_style_transfer/transformation/einsum/transpose5fast_style_transfer/transformation/einsum/transpose_1*"
_output_shapes
:  *
T0

:fast_style_transfer/transformation/einsum/transpose_2/permConst*
_output_shapes
:*!
valueB"          *
dtype0
Ý
5fast_style_transfer/transformation/einsum/transpose_2	Transpose0fast_style_transfer/transformation/einsum/MatMul:fast_style_transfer/transformation/einsum/transpose_2/perm*
T0*"
_output_shapes
:  

+fast_style_transfer/compressor/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
«
"fast_style_transfer/compressor/PadPadvgg19_encoder/block3_conv1/Relu+fast_style_transfer/compressor/Pad/paddings*
T0*)
_output_shapes
:
û
Wfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*
dtype0*
_output_shapes
:
å
Ufast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *:Í½*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*
dtype0*
_output_shapes
: 
å
Ufast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *:Í=*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*
dtype0*
_output_shapes
: 
Ô
_fast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniformWfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel
ö
Ufast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/subSubUfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/maxUfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel

Ufast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/mulMul_fast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/RandomUniformUfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/sub*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*(
_output_shapes
:

Qfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniformAddUfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/mulUfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*(
_output_shapes
:
å
6fast_style_transfer/compressor/compressor/conv1/kernel
VariableV2*
shape:*
dtype0*(
_output_shapes
:*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel
Ð
=fast_style_transfer/compressor/compressor/conv1/kernel/AssignAssign6fast_style_transfer/compressor/compressor/conv1/kernelQfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*(
_output_shapes
:
ý
;fast_style_transfer/compressor/compressor/conv1/kernel/readIdentity6fast_style_transfer/compressor/compressor/conv1/kernel*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*(
_output_shapes
:*
T0
Þ
Ffast_style_transfer/compressor/compressor/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv1/bias
Ç
4fast_style_transfer/compressor/compressor/conv1/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv1/bias
²
;fast_style_transfer/compressor/compressor/conv1/bias/AssignAssign4fast_style_transfer/compressor/compressor/conv1/biasFfast_style_transfer/compressor/compressor/conv1/bias/Initializer/zeros*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv1/bias*
_output_shapes	
:
ê
9fast_style_transfer/compressor/compressor/conv1/bias/readIdentity4fast_style_transfer/compressor/compressor/conv1/bias*
_output_shapes	
:*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv1/bias

=fast_style_transfer/compressor/compressor/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
þ
6fast_style_transfer/compressor/compressor/conv1/Conv2DConv2D"fast_style_transfer/compressor/Pad;fast_style_transfer/compressor/compressor/conv1/kernel/read*
paddingVALID*)
_output_shapes
:*
T0*
strides

é
7fast_style_transfer/compressor/compressor/conv1/BiasAddBiasAdd6fast_style_transfer/compressor/compressor/conv1/Conv2D9fast_style_transfer/compressor/compressor/conv1/bias/read*
T0*)
_output_shapes
:
©
4fast_style_transfer/compressor/compressor/conv1/ReluRelu7fast_style_transfer/compressor/compressor/conv1/BiasAdd*)
_output_shapes
:*
T0

-fast_style_transfer/compressor/Pad_1/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
Ä
$fast_style_transfer/compressor/Pad_1Pad4fast_style_transfer/compressor/compressor/conv1/Relu-fast_style_transfer/compressor/Pad_1/paddings*)
_output_shapes
:*
T0
û
Wfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*
dtype0*
_output_shapes
:
å
Ufast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*
dtype0*
_output_shapes
: 
å
Ufast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ª*=*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*
dtype0*
_output_shapes
: 
Ô
_fast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniformWfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel
ö
Ufast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/subSubUfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/maxUfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*
_output_shapes
: 

Ufast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/mulMul_fast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/RandomUniformUfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/sub*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*(
_output_shapes
:

Qfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniformAddUfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/mulUfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*(
_output_shapes
:
å
6fast_style_transfer/compressor/compressor/conv2/kernel
VariableV2*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*
shape:*
dtype0*(
_output_shapes
:
Ð
=fast_style_transfer/compressor/compressor/conv2/kernel/AssignAssign6fast_style_transfer/compressor/compressor/conv2/kernelQfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*(
_output_shapes
:
ý
;fast_style_transfer/compressor/compressor/conv2/kernel/readIdentity6fast_style_transfer/compressor/compressor/conv2/kernel*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*(
_output_shapes
:
Þ
Ffast_style_transfer/compressor/compressor/conv2/bias/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv2/bias*
dtype0*
_output_shapes	
:
Ç
4fast_style_transfer/compressor/compressor/conv2/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv2/bias
²
;fast_style_transfer/compressor/compressor/conv2/bias/AssignAssign4fast_style_transfer/compressor/compressor/conv2/biasFfast_style_transfer/compressor/compressor/conv2/bias/Initializer/zeros*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv2/bias*
_output_shapes	
:
ê
9fast_style_transfer/compressor/compressor/conv2/bias/readIdentity4fast_style_transfer/compressor/compressor/conv2/bias*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv2/bias*
_output_shapes	
:

=fast_style_transfer/compressor/compressor/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

6fast_style_transfer/compressor/compressor/conv2/Conv2DConv2D$fast_style_transfer/compressor/Pad_1;fast_style_transfer/compressor/compressor/conv2/kernel/read*)
_output_shapes
:*
T0*
strides
*
paddingVALID
é
7fast_style_transfer/compressor/compressor/conv2/BiasAddBiasAdd6fast_style_transfer/compressor/compressor/conv2/Conv2D9fast_style_transfer/compressor/compressor/conv2/bias/read*)
_output_shapes
:*
T0
©
4fast_style_transfer/compressor/compressor/conv2/ReluRelu7fast_style_transfer/compressor/compressor/conv2/BiasAdd*)
_output_shapes
:*
T0

-fast_style_transfer/compressor/Pad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ä
$fast_style_transfer/compressor/Pad_2Pad4fast_style_transfer/compressor/compressor/conv2/Relu-fast_style_transfer/compressor/Pad_2/paddings*)
_output_shapes
:*
T0
û
Wfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*
dtype0*
_output_shapes
:
å
Ufast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *¥2½*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*
dtype0*
_output_shapes
: 
å
Ufast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *¥2=*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel
Ó
_fast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniformWfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/shape*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*
dtype0*'
_output_shapes
: 
ö
Ufast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/subSubUfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/maxUfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*
_output_shapes
: 

Ufast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/mulMul_fast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/RandomUniformUfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/sub*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*'
_output_shapes
: 

Qfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniformAddUfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/mulUfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform/min*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*'
_output_shapes
: 
ã
6fast_style_transfer/compressor/compressor/conv3/kernel
VariableV2*
shape: *
dtype0*'
_output_shapes
: *I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel
Ï
=fast_style_transfer/compressor/compressor/conv3/kernel/AssignAssign6fast_style_transfer/compressor/compressor/conv3/kernelQfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*'
_output_shapes
: 
ü
;fast_style_transfer/compressor/compressor/conv3/kernel/readIdentity6fast_style_transfer/compressor/compressor/conv3/kernel*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*'
_output_shapes
: *
T0
Ü
Ffast_style_transfer/compressor/compressor/conv3/bias/Initializer/zerosConst*
valueB *    *G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv3/bias*
dtype0*
_output_shapes
: 
Å
4fast_style_transfer/compressor/compressor/conv3/bias
VariableV2*
dtype0*
_output_shapes
: *G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv3/bias*
shape: 
±
;fast_style_transfer/compressor/compressor/conv3/bias/AssignAssign4fast_style_transfer/compressor/compressor/conv3/biasFfast_style_transfer/compressor/compressor/conv3/bias/Initializer/zeros*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv3/bias*
_output_shapes
: 
é
9fast_style_transfer/compressor/compressor/conv3/bias/readIdentity4fast_style_transfer/compressor/compressor/conv3/bias*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv3/bias*
_output_shapes
: 

=fast_style_transfer/compressor/compressor/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ÿ
6fast_style_transfer/compressor/compressor/conv3/Conv2DConv2D$fast_style_transfer/compressor/Pad_2;fast_style_transfer/compressor/compressor/conv3/kernel/read*
paddingVALID*(
_output_shapes
: *
T0*
strides

è
7fast_style_transfer/compressor/compressor/conv3/BiasAddBiasAdd6fast_style_transfer/compressor/compressor/conv3/Conv2D9fast_style_transfer/compressor/compressor/conv3/bias/read*(
_output_shapes
: *
T0
¨
4fast_style_transfer/compressor/compressor/conv3/ReluRelu7fast_style_transfer/compressor/compressor/conv3/BiasAdd*
T0*(
_output_shapes
: 
}
$fast_style_transfer/compressor/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"             
|
2fast_style_transfer/compressor/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
~
4fast_style_transfer/compressor/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
~
4fast_style_transfer/compressor/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
È
,fast_style_transfer/compressor/strided_sliceStridedSlice$fast_style_transfer/compressor/Shape2fast_style_transfer/compressor/strided_slice/stack4fast_style_transfer/compressor/strided_slice/stack_14fast_style_transfer/compressor/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
y
.fast_style_transfer/compressor/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
p
.fast_style_transfer/compressor/Reshape/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
ð
,fast_style_transfer/compressor/Reshape/shapePack,fast_style_transfer/compressor/strided_slice.fast_style_transfer/compressor/Reshape/shape/1.fast_style_transfer/compressor/Reshape/shape/2*
N*
_output_shapes
:*
T0
Ä
&fast_style_transfer/compressor/ReshapeReshape4fast_style_transfer/compressor/compressor/conv3/Relu,fast_style_transfer/compressor/Reshape/shape*$
_output_shapes
: *
T0

-fast_style_transfer/compressor_1/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
±
$fast_style_transfer/compressor_1/PadPad!vgg19_encoder_1/block3_conv1/Relu-fast_style_transfer/compressor_1/Pad/paddings*
T0*)
_output_shapes
:

?fast_style_transfer/compressor_1/compressor/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

8fast_style_transfer/compressor_1/compressor/conv1/Conv2DConv2D$fast_style_transfer/compressor_1/Pad;fast_style_transfer/compressor/compressor/conv1/kernel/read*
T0*
strides
*
paddingVALID*)
_output_shapes
:
í
9fast_style_transfer/compressor_1/compressor/conv1/BiasAddBiasAdd8fast_style_transfer/compressor_1/compressor/conv1/Conv2D9fast_style_transfer/compressor/compressor/conv1/bias/read*
T0*)
_output_shapes
:
­
6fast_style_transfer/compressor_1/compressor/conv1/ReluRelu9fast_style_transfer/compressor_1/compressor/conv1/BiasAdd*
T0*)
_output_shapes
:
 
/fast_style_transfer/compressor_1/Pad_1/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ê
&fast_style_transfer/compressor_1/Pad_1Pad6fast_style_transfer/compressor_1/compressor/conv1/Relu/fast_style_transfer/compressor_1/Pad_1/paddings*)
_output_shapes
:*
T0

?fast_style_transfer/compressor_1/compressor/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      

8fast_style_transfer/compressor_1/compressor/conv2/Conv2DConv2D&fast_style_transfer/compressor_1/Pad_1;fast_style_transfer/compressor/compressor/conv2/kernel/read*
strides
*
paddingVALID*)
_output_shapes
:*
T0
í
9fast_style_transfer/compressor_1/compressor/conv2/BiasAddBiasAdd8fast_style_transfer/compressor_1/compressor/conv2/Conv2D9fast_style_transfer/compressor/compressor/conv2/bias/read*
T0*)
_output_shapes
:
­
6fast_style_transfer/compressor_1/compressor/conv2/ReluRelu9fast_style_transfer/compressor_1/compressor/conv2/BiasAdd*
T0*)
_output_shapes
:
 
/fast_style_transfer/compressor_1/Pad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ê
&fast_style_transfer/compressor_1/Pad_2Pad6fast_style_transfer/compressor_1/compressor/conv2/Relu/fast_style_transfer/compressor_1/Pad_2/paddings*)
_output_shapes
:*
T0

?fast_style_transfer/compressor_1/compressor/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

8fast_style_transfer/compressor_1/compressor/conv3/Conv2DConv2D&fast_style_transfer/compressor_1/Pad_2;fast_style_transfer/compressor/compressor/conv3/kernel/read*(
_output_shapes
: *
T0*
strides
*
paddingVALID
ì
9fast_style_transfer/compressor_1/compressor/conv3/BiasAddBiasAdd8fast_style_transfer/compressor_1/compressor/conv3/Conv2D9fast_style_transfer/compressor/compressor/conv3/bias/read*(
_output_shapes
: *
T0
¬
6fast_style_transfer/compressor_1/compressor/conv3/ReluRelu9fast_style_transfer/compressor_1/compressor/conv3/BiasAdd*(
_output_shapes
: *
T0

&fast_style_transfer/compressor_1/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
~
4fast_style_transfer/compressor_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

6fast_style_transfer/compressor_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

6fast_style_transfer/compressor_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ò
.fast_style_transfer/compressor_1/strided_sliceStridedSlice&fast_style_transfer/compressor_1/Shape4fast_style_transfer/compressor_1/strided_slice/stack6fast_style_transfer/compressor_1/strided_slice/stack_16fast_style_transfer/compressor_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
{
0fast_style_transfer/compressor_1/Reshape/shape/1Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
r
0fast_style_transfer/compressor_1/Reshape/shape/2Const*
value	B : *
dtype0*
_output_shapes
: 
ø
.fast_style_transfer/compressor_1/Reshape/shapePack.fast_style_transfer/compressor_1/strided_slice0fast_style_transfer/compressor_1/Reshape/shape/10fast_style_transfer/compressor_1/Reshape/shape/2*
N*
_output_shapes
:*
T0
Ê
(fast_style_transfer/compressor_1/ReshapeReshape6fast_style_transfer/compressor_1/compressor/conv3/Relu.fast_style_transfer/compressor_1/Reshape/shape*
T0*$
_output_shapes
: 
|
2fast_style_transfer/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Â
 fast_style_transfer/moments/meanMean&fast_style_transfer/compressor/Reshape2fast_style_transfer/moments/mean/reduction_indices*
T0*"
_output_shapes
: *
	keep_dims(

(fast_style_transfer/moments/StopGradientStopGradient fast_style_transfer/moments/mean*"
_output_shapes
: *
T0
Ã
-fast_style_transfer/moments/SquaredDifferenceSquaredDifference&fast_style_transfer/compressor/Reshape(fast_style_transfer/moments/StopGradient*
T0*$
_output_shapes
: 

6fast_style_transfer/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ñ
$fast_style_transfer/moments/varianceMean-fast_style_transfer/moments/SquaredDifference6fast_style_transfer/moments/variance/reduction_indices*
T0*"
_output_shapes
: *
	keep_dims(

fast_style_transfer/subSub&fast_style_transfer/compressor/Reshape fast_style_transfer/moments/mean*$
_output_shapes
: *
T0
¨
fast_style_transfer/MatMulBatchMatMulfast_style_transfer/sub5fast_style_transfer/transformation/einsum/transpose_2*$
_output_shapes
: *
T0
~
4fast_style_transfer/moments_1/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
È
"fast_style_transfer/moments_1/meanMean(fast_style_transfer/compressor_1/Reshape4fast_style_transfer/moments_1/mean/reduction_indices*
T0*"
_output_shapes
: *
	keep_dims(

*fast_style_transfer/moments_1/StopGradientStopGradient"fast_style_transfer/moments_1/mean*
T0*"
_output_shapes
: 
É
/fast_style_transfer/moments_1/SquaredDifferenceSquaredDifference(fast_style_transfer/compressor_1/Reshape*fast_style_transfer/moments_1/StopGradient*
T0*$
_output_shapes
: 

8fast_style_transfer/moments_1/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
×
&fast_style_transfer/moments_1/varianceMean/fast_style_transfer/moments_1/SquaredDifference8fast_style_transfer/moments_1/variance/reduction_indices*"
_output_shapes
: *
	keep_dims(*
T0

fast_style_transfer/addAddfast_style_transfer/MatMul"fast_style_transfer/moments_1/mean*
T0*$
_output_shapes
: 
n
fast_style_transfer/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"    @      
q
'fast_style_transfer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)fast_style_transfer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)fast_style_transfer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

!fast_style_transfer/strided_sliceStridedSlicefast_style_transfer/Shape'fast_style_transfer/strided_slice/stack)fast_style_transfer/strided_slice/stack_1)fast_style_transfer/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
s
)fast_style_transfer/strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
u
+fast_style_transfer/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+fast_style_transfer/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

#fast_style_transfer/strided_slice_1StridedSliceShape)fast_style_transfer/strided_slice_1/stack+fast_style_transfer/strided_slice_1/stack_1+fast_style_transfer/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
_
fast_style_transfer/truediv/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
 fast_style_transfer/truediv/CastCast#fast_style_transfer/strided_slice_1*

SrcT0*

DstT0*
_output_shapes
: 
y
"fast_style_transfer/truediv/Cast_1Castfast_style_transfer/truediv/y*

SrcT0*

DstT0*
_output_shapes
: 

fast_style_transfer/truedivRealDiv fast_style_transfer/truediv/Cast"fast_style_transfer/truediv/Cast_1*
T0*
_output_shapes
: 
s
)fast_style_transfer/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
u
+fast_style_transfer/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+fast_style_transfer/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

#fast_style_transfer/strided_slice_2StridedSliceShape)fast_style_transfer/strided_slice_2/stack+fast_style_transfer/strided_slice_2/stack_1+fast_style_transfer/strided_slice_2/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
a
fast_style_transfer/truediv_1/yConst*
value	B :*
dtype0*
_output_shapes
: 

"fast_style_transfer/truediv_1/CastCast#fast_style_transfer/strided_slice_2*

SrcT0*

DstT0*
_output_shapes
: 
}
$fast_style_transfer/truediv_1/Cast_1Castfast_style_transfer/truediv_1/y*

SrcT0*

DstT0*
_output_shapes
: 

fast_style_transfer/truediv_1RealDiv"fast_style_transfer/truediv_1/Cast$fast_style_transfer/truediv_1/Cast_1*
T0*
_output_shapes
: 

-fast_style_transfer/uncompressor/Reshape/CastCastfast_style_transfer/truediv*

SrcT0*

DstT0*
_output_shapes
: 

/fast_style_transfer/uncompressor/Reshape/Cast_1Castfast_style_transfer/truediv_1*

SrcT0*

DstT0*
_output_shapes
: 
{
0fast_style_transfer/uncompressor/Reshape/shape/0Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
r
0fast_style_transfer/uncompressor/Reshape/shape/3Const*
value	B : *
dtype0*
_output_shapes
: 
¨
.fast_style_transfer/uncompressor/Reshape/shapePack0fast_style_transfer/uncompressor/Reshape/shape/0-fast_style_transfer/uncompressor/Reshape/Cast/fast_style_transfer/uncompressor/Reshape/Cast_10fast_style_transfer/uncompressor/Reshape/shape/3*
T0*
N*
_output_shapes
:
¯
(fast_style_transfer/uncompressor/ReshapeReshapefast_style_transfer/add.fast_style_transfer/uncompressor/Reshape/shape*(
_output_shapes
: *
T0

-fast_style_transfer/uncompressor/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
·
$fast_style_transfer/uncompressor/PadPad(fast_style_transfer/uncompressor/Reshape-fast_style_transfer/uncompressor/Pad/paddings*
T0*(
_output_shapes
: 

[fast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"          @   *M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
dtype0
í
Yfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *«ªª½*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
dtype0*
_output_shapes
: 
í
Yfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ªª=*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
dtype0*
_output_shapes
: 
Þ
cfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform[fast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/shape*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
dtype0*&
_output_shapes
: @

Yfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/subSubYfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/maxYfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel
 
Yfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/mulMulcfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/RandomUniformYfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/sub*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*&
_output_shapes
: @*
T0

Ufast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniformAddYfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/mulYfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform/min*&
_output_shapes
: @*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel
é
:fast_style_transfer/uncompressor/uncompressor/conv1/kernel
VariableV2*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
shape: @*
dtype0*&
_output_shapes
: @
Þ
Afast_style_transfer/uncompressor/uncompressor/conv1/kernel/AssignAssign:fast_style_transfer/uncompressor/uncompressor/conv1/kernelUfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*&
_output_shapes
: @

?fast_style_transfer/uncompressor/uncompressor/conv1/kernel/readIdentity:fast_style_transfer/uncompressor/uncompressor/conv1/kernel*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*&
_output_shapes
: @
ä
Jfast_style_transfer/uncompressor/uncompressor/conv1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv1/bias
Í
8fast_style_transfer/uncompressor/uncompressor/conv1/bias
VariableV2*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv1/bias*
shape:@*
dtype0*
_output_shapes
:@
Á
?fast_style_transfer/uncompressor/uncompressor/conv1/bias/AssignAssign8fast_style_transfer/uncompressor/uncompressor/conv1/biasJfast_style_transfer/uncompressor/uncompressor/conv1/bias/Initializer/zeros*
_output_shapes
:@*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv1/bias
õ
=fast_style_transfer/uncompressor/uncompressor/conv1/bias/readIdentity8fast_style_transfer/uncompressor/uncompressor/conv1/bias*
_output_shapes
:@*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv1/bias

Afast_style_transfer/uncompressor/uncompressor/conv1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0

:fast_style_transfer/uncompressor/uncompressor/conv1/Conv2DConv2D$fast_style_transfer/uncompressor/Pad?fast_style_transfer/uncompressor/uncompressor/conv1/kernel/read*(
_output_shapes
:@*
T0*
strides
*
paddingVALID
ô
;fast_style_transfer/uncompressor/uncompressor/conv1/BiasAddBiasAdd:fast_style_transfer/uncompressor/uncompressor/conv1/Conv2D=fast_style_transfer/uncompressor/uncompressor/conv1/bias/read*(
_output_shapes
:@*
T0
°
8fast_style_transfer/uncompressor/uncompressor/conv1/ReluRelu;fast_style_transfer/uncompressor/uncompressor/conv1/BiasAdd*(
_output_shapes
:@*
T0
 
/fast_style_transfer/uncompressor/Pad_1/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
Ë
&fast_style_transfer/uncompressor/Pad_1Pad8fast_style_transfer/uncompressor/uncompressor/conv1/Relu/fast_style_transfer/uncompressor/Pad_1/paddings*
T0*(
_output_shapes
:@

[fast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @      *M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*
dtype0*
_output_shapes
:
í
Yfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ï[q½*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel
í
Yfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ï[q=*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*
dtype0*
_output_shapes
: 
ß
cfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform[fast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/shape*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*
dtype0*'
_output_shapes
:@*
T0

Yfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/subSubYfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/maxYfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel
¡
Yfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/mulMulcfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/RandomUniformYfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/sub*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*'
_output_shapes
:@

Ufast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniformAddYfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/mulYfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform/min*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*'
_output_shapes
:@
ë
:fast_style_transfer/uncompressor/uncompressor/conv2/kernel
VariableV2*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*
shape:@*
dtype0*'
_output_shapes
:@
ß
Afast_style_transfer/uncompressor/uncompressor/conv2/kernel/AssignAssign:fast_style_transfer/uncompressor/uncompressor/conv2/kernelUfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform*'
_output_shapes
:@*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel

?fast_style_transfer/uncompressor/uncompressor/conv2/kernel/readIdentity:fast_style_transfer/uncompressor/uncompressor/conv2/kernel*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*'
_output_shapes
:@
æ
Jfast_style_transfer/uncompressor/uncompressor/conv2/bias/Initializer/zerosConst*
valueB*    *K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv2/bias*
dtype0*
_output_shapes	
:
Ï
8fast_style_transfer/uncompressor/uncompressor/conv2/bias
VariableV2*
dtype0*
_output_shapes	
:*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv2/bias*
shape:
Â
?fast_style_transfer/uncompressor/uncompressor/conv2/bias/AssignAssign8fast_style_transfer/uncompressor/uncompressor/conv2/biasJfast_style_transfer/uncompressor/uncompressor/conv2/bias/Initializer/zeros*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv2/bias*
_output_shapes	
:*
T0
ö
=fast_style_transfer/uncompressor/uncompressor/conv2/bias/readIdentity8fast_style_transfer/uncompressor/uncompressor/conv2/bias*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv2/bias*
_output_shapes	
:

Afast_style_transfer/uncompressor/uncompressor/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

:fast_style_transfer/uncompressor/uncompressor/conv2/Conv2DConv2D&fast_style_transfer/uncompressor/Pad_1?fast_style_transfer/uncompressor/uncompressor/conv2/kernel/read*
paddingVALID*)
_output_shapes
:*
T0*
strides

õ
;fast_style_transfer/uncompressor/uncompressor/conv2/BiasAddBiasAdd:fast_style_transfer/uncompressor/uncompressor/conv2/Conv2D=fast_style_transfer/uncompressor/uncompressor/conv2/bias/read*
T0*)
_output_shapes
:
±
8fast_style_transfer/uncompressor/uncompressor/conv2/ReluRelu;fast_style_transfer/uncompressor/uncompressor/conv2/BiasAdd*
T0*)
_output_shapes
:
 
/fast_style_transfer/uncompressor/Pad_2/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
Ì
&fast_style_transfer/uncompressor/Pad_2Pad8fast_style_transfer/uncompressor/uncompressor/conv2/Relu/fast_style_transfer/uncompressor/Pad_2/paddings*
T0*)
_output_shapes
:

[fast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*
dtype0*
_output_shapes
:
í
Yfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*
dtype0*
_output_shapes
: 
í
Yfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ª*=*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*
dtype0*
_output_shapes
: 
à
cfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniform[fast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel

Yfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/subSubYfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/maxYfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel
¢
Yfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/mulMulcfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/RandomUniformYfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/sub*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*(
_output_shapes
:

Ufast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniformAddYfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/mulYfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform/min*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*(
_output_shapes
:*
T0
í
:fast_style_transfer/uncompressor/uncompressor/conv3/kernel
VariableV2*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*
shape:*
dtype0*(
_output_shapes
:
à
Afast_style_transfer/uncompressor/uncompressor/conv3/kernel/AssignAssign:fast_style_transfer/uncompressor/uncompressor/conv3/kernelUfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*(
_output_shapes
:

?fast_style_transfer/uncompressor/uncompressor/conv3/kernel/readIdentity:fast_style_transfer/uncompressor/uncompressor/conv3/kernel*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*(
_output_shapes
:
æ
Jfast_style_transfer/uncompressor/uncompressor/conv3/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv3/bias*
dtype0
Ï
8fast_style_transfer/uncompressor/uncompressor/conv3/bias
VariableV2*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv3/bias*
shape:*
dtype0*
_output_shapes	
:
Â
?fast_style_transfer/uncompressor/uncompressor/conv3/bias/AssignAssign8fast_style_transfer/uncompressor/uncompressor/conv3/biasJfast_style_transfer/uncompressor/uncompressor/conv3/bias/Initializer/zeros*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv3/bias*
_output_shapes	
:
ö
=fast_style_transfer/uncompressor/uncompressor/conv3/bias/readIdentity8fast_style_transfer/uncompressor/uncompressor/conv3/bias*
_output_shapes	
:*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv3/bias

Afast_style_transfer/uncompressor/uncompressor/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

:fast_style_transfer/uncompressor/uncompressor/conv3/Conv2DConv2D&fast_style_transfer/uncompressor/Pad_2?fast_style_transfer/uncompressor/uncompressor/conv3/kernel/read*
paddingVALID*)
_output_shapes
:*
T0*
strides

õ
;fast_style_transfer/uncompressor/uncompressor/conv3/BiasAddBiasAdd:fast_style_transfer/uncompressor/uncompressor/conv3/Conv2D=fast_style_transfer/uncompressor/uncompressor/conv3/bias/read*)
_output_shapes
:*
T0
±
8fast_style_transfer/uncompressor/uncompressor/conv3/ReluRelu;fast_style_transfer/uncompressor/uncompressor/conv3/BiasAdd*)
_output_shapes
:*
T0

vgg19_decoder/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
¢
vgg19_decoder/PadPad8fast_style_transfer/uncompressor/uncompressor/conv3/Reluvgg19_decoder/Pad/paddings*
T0*)
_output_shapes
:
á
Jvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*
dtype0
Ë
Hvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/minConst*
valueB
 *«ª*½*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*
dtype0*
_output_shapes
: 
Ë
Hvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/maxConst*
valueB
 *«ª*=*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*
dtype0*
_output_shapes
: 
­
Rvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformRandomUniformJvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel
Â
Hvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/subSubHvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/maxHvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel
Þ
Hvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/mulMulRvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformHvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/sub*(
_output_shapes
:*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel
Ð
Dvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniformAddHvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/mulHvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform/min*(
_output_shapes
:*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel
Ë
)vgg19_decoder/block3_conv1_decoder/kernel
VariableV2*
shape:*
dtype0*(
_output_shapes
:*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel

0vgg19_decoder/block3_conv1_decoder/kernel/AssignAssign)vgg19_decoder/block3_conv1_decoder/kernelDvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*(
_output_shapes
:
Ö
.vgg19_decoder/block3_conv1_decoder/kernel/readIdentity)vgg19_decoder/block3_conv1_decoder/kernel*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*(
_output_shapes
:
Ä
9vgg19_decoder/block3_conv1_decoder/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *:
_class0
.,loc:@vgg19_decoder/block3_conv1_decoder/bias
­
'vgg19_decoder/block3_conv1_decoder/bias
VariableV2*
dtype0*
_output_shapes	
:*:
_class0
.,loc:@vgg19_decoder/block3_conv1_decoder/bias*
shape:
þ
.vgg19_decoder/block3_conv1_decoder/bias/AssignAssign'vgg19_decoder/block3_conv1_decoder/bias9vgg19_decoder/block3_conv1_decoder/bias/Initializer/zeros*
T0*:
_class0
.,loc:@vgg19_decoder/block3_conv1_decoder/bias*
_output_shapes	
:
Ã
,vgg19_decoder/block3_conv1_decoder/bias/readIdentity'vgg19_decoder/block3_conv1_decoder/bias*
_output_shapes	
:*
T0*:
_class0
.,loc:@vgg19_decoder/block3_conv1_decoder/bias

0vgg19_decoder/block3_conv1_decoder/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ó
)vgg19_decoder/block3_conv1_decoder/Conv2DConv2Dvgg19_decoder/Pad.vgg19_decoder/block3_conv1_decoder/kernel/read*
T0*
strides
*
paddingVALID*)
_output_shapes
:
Â
*vgg19_decoder/block3_conv1_decoder/BiasAddBiasAdd)vgg19_decoder/block3_conv1_decoder/Conv2D,vgg19_decoder/block3_conv1_decoder/bias/read*)
_output_shapes
:*
T0

'vgg19_decoder/block3_conv1_decoder/ReluRelu*vgg19_decoder/block3_conv1_decoder/BiasAdd*
T0*)
_output_shapes
:

.vgg19_decoder/nearest_upsampling/Reshape/shapeConst*-
value$B""                  *
dtype0*
_output_shapes
:
È
(vgg19_decoder/nearest_upsampling/ReshapeReshape'vgg19_decoder/block3_conv1_decoder/Relu.vgg19_decoder/nearest_upsampling/Reshape/shape*1
_output_shapes
:*
T0

%vgg19_decoder/nearest_upsampling/onesConst*-
value$B"*  ?*
dtype0*.
_output_shapes
:
¸
$vgg19_decoder/nearest_upsampling/mulMul(vgg19_decoder/nearest_upsampling/Reshape%vgg19_decoder/nearest_upsampling/ones*1
_output_shapes
:*
T0

0vgg19_decoder/nearest_upsampling/Reshape_1/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Á
*vgg19_decoder/nearest_upsampling/Reshape_1Reshape$vgg19_decoder/nearest_upsampling/mul0vgg19_decoder/nearest_upsampling/Reshape_1/shape*
T0*)
_output_shapes
:

vgg19_decoder/Pad_1/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:

vgg19_decoder/Pad_1Pad*vgg19_decoder/nearest_upsampling/Reshape_1vgg19_decoder/Pad_1/paddings*
T0*)
_output_shapes
:
õ
Tvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*
dtype0*
_output_shapes
:
ß
Rvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/minConst*
valueB
 *ìQ½*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*
dtype0*
_output_shapes
: 
ß
Rvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ìQ=*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel
Ë
\vgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/RandomUniformRandomUniformTvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/shape*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*
dtype0*(
_output_shapes
:
ê
Rvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/subSubRvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/maxRvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/min*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*
_output_shapes
: *
T0

Rvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/mulMul\vgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/RandomUniformRvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*(
_output_shapes
:
ø
Nvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniformAddRvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/mulRvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*(
_output_shapes
:
ß
3vgg19_decoder/block2_conv2_transpose_decoder/kernel
VariableV2*
dtype0*(
_output_shapes
:*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*
shape:
Ä
:vgg19_decoder/block2_conv2_transpose_decoder/kernel/AssignAssign3vgg19_decoder/block2_conv2_transpose_decoder/kernelNvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform*(
_output_shapes
:*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel
ô
8vgg19_decoder/block2_conv2_transpose_decoder/kernel/readIdentity3vgg19_decoder/block2_conv2_transpose_decoder/kernel*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*(
_output_shapes
:
Ø
Cvgg19_decoder/block2_conv2_transpose_decoder/bias/Initializer/zerosConst*
valueB*    *D
_class:
86loc:@vgg19_decoder/block2_conv2_transpose_decoder/bias*
dtype0*
_output_shapes	
:
Á
1vgg19_decoder/block2_conv2_transpose_decoder/bias
VariableV2*D
_class:
86loc:@vgg19_decoder/block2_conv2_transpose_decoder/bias*
shape:*
dtype0*
_output_shapes	
:
¦
8vgg19_decoder/block2_conv2_transpose_decoder/bias/AssignAssign1vgg19_decoder/block2_conv2_transpose_decoder/biasCvgg19_decoder/block2_conv2_transpose_decoder/bias/Initializer/zeros*
T0*D
_class:
86loc:@vgg19_decoder/block2_conv2_transpose_decoder/bias*
_output_shapes	
:
á
6vgg19_decoder/block2_conv2_transpose_decoder/bias/readIdentity1vgg19_decoder/block2_conv2_transpose_decoder/bias*
T0*D
_class:
86loc:@vgg19_decoder/block2_conv2_transpose_decoder/bias*
_output_shapes	
:

:vgg19_decoder/block2_conv2_transpose_decoder/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
é
3vgg19_decoder/block2_conv2_transpose_decoder/Conv2DConv2Dvgg19_decoder/Pad_18vgg19_decoder/block2_conv2_transpose_decoder/kernel/read*
strides
*
paddingVALID*)
_output_shapes
:*
T0
à
4vgg19_decoder/block2_conv2_transpose_decoder/BiasAddBiasAdd3vgg19_decoder/block2_conv2_transpose_decoder/Conv2D6vgg19_decoder/block2_conv2_transpose_decoder/bias/read*
T0*)
_output_shapes
:
£
1vgg19_decoder/block2_conv2_transpose_decoder/ReluRelu4vgg19_decoder/block2_conv2_transpose_decoder/BiasAdd*
T0*)
_output_shapes
:

vgg19_decoder/Pad_2/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:

vgg19_decoder/Pad_2Pad1vgg19_decoder/block2_conv2_transpose_decoder/Reluvgg19_decoder/Pad_2/paddings*
T0*)
_output_shapes
:
á
Jvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*
dtype0*
_output_shapes
:
Ë
Hvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ï[q½*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*
dtype0
Ë
Hvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/maxConst*
valueB
 *ï[q=*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*
dtype0*
_output_shapes
: 
¬
Rvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformRandomUniformJvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/shape*
dtype0*'
_output_shapes
:@*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel
Â
Hvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/subSubHvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/maxHvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*
_output_shapes
: 
Ý
Hvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/mulMulRvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformHvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/sub*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*'
_output_shapes
:@
Ï
Dvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniformAddHvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/mulHvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*'
_output_shapes
:@
É
)vgg19_decoder/block2_conv1_decoder/kernel
VariableV2*
dtype0*'
_output_shapes
:@*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*
shape:@

0vgg19_decoder/block2_conv1_decoder/kernel/AssignAssign)vgg19_decoder/block2_conv1_decoder/kernelDvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform*'
_output_shapes
:@*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel
Õ
.vgg19_decoder/block2_conv1_decoder/kernel/readIdentity)vgg19_decoder/block2_conv1_decoder/kernel*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*'
_output_shapes
:@*
T0
Â
9vgg19_decoder/block2_conv1_decoder/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *:
_class0
.,loc:@vgg19_decoder/block2_conv1_decoder/bias
«
'vgg19_decoder/block2_conv1_decoder/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*:
_class0
.,loc:@vgg19_decoder/block2_conv1_decoder/bias
ý
.vgg19_decoder/block2_conv1_decoder/bias/AssignAssign'vgg19_decoder/block2_conv1_decoder/bias9vgg19_decoder/block2_conv1_decoder/bias/Initializer/zeros*
T0*:
_class0
.,loc:@vgg19_decoder/block2_conv1_decoder/bias*
_output_shapes
:@
Â
,vgg19_decoder/block2_conv1_decoder/bias/readIdentity'vgg19_decoder/block2_conv1_decoder/bias*
_output_shapes
:@*
T0*:
_class0
.,loc:@vgg19_decoder/block2_conv1_decoder/bias

0vgg19_decoder/block2_conv1_decoder/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ô
)vgg19_decoder/block2_conv1_decoder/Conv2DConv2Dvgg19_decoder/Pad_2.vgg19_decoder/block2_conv1_decoder/kernel/read*
paddingVALID*(
_output_shapes
:@*
T0*
strides

Á
*vgg19_decoder/block2_conv1_decoder/BiasAddBiasAdd)vgg19_decoder/block2_conv1_decoder/Conv2D,vgg19_decoder/block2_conv1_decoder/bias/read*
T0*(
_output_shapes
:@

'vgg19_decoder/block2_conv1_decoder/ReluRelu*vgg19_decoder/block2_conv1_decoder/BiasAdd*
T0*(
_output_shapes
:@

0vgg19_decoder/nearest_upsampling_1/Reshape/shapeConst*
_output_shapes
:*-
value$B""               @   *
dtype0
Ë
*vgg19_decoder/nearest_upsampling_1/ReshapeReshape'vgg19_decoder/block2_conv1_decoder/Relu0vgg19_decoder/nearest_upsampling_1/Reshape/shape*0
_output_shapes
:@*
T0

'vgg19_decoder/nearest_upsampling_1/onesConst*.
_output_shapes
:*-
value$B"*  ?*
dtype0
½
&vgg19_decoder/nearest_upsampling_1/mulMul*vgg19_decoder/nearest_upsampling_1/Reshape'vgg19_decoder/nearest_upsampling_1/ones*0
_output_shapes
:@*
T0

2vgg19_decoder/nearest_upsampling_1/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Æ
,vgg19_decoder/nearest_upsampling_1/Reshape_1Reshape&vgg19_decoder/nearest_upsampling_1/mul2vgg19_decoder/nearest_upsampling_1/Reshape_1/shape*
T0*(
_output_shapes
:@

vgg19_decoder/Pad_3/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0

vgg19_decoder/Pad_3Pad,vgg19_decoder/nearest_upsampling_1/Reshape_1vgg19_decoder/Pad_3/paddings*(
_output_shapes
:@*
T0
õ
Tvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*
dtype0*
_output_shapes
:
ß
Rvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/minConst*
valueB
 *:Í½*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*
dtype0*
_output_shapes
: 
ß
Rvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:Í=*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*
dtype0
É
\vgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/RandomUniformRandomUniformTvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@@*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel
ê
Rvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/subSubRvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/maxRvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*
_output_shapes
: 

Rvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/mulMul\vgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/RandomUniformRvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*&
_output_shapes
:@@
ö
Nvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniformAddRvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/mulRvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform/min*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*&
_output_shapes
:@@*
T0
Û
3vgg19_decoder/block1_conv2_transpose_decoder/kernel
VariableV2*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*
shape:@@*
dtype0*&
_output_shapes
:@@
Â
:vgg19_decoder/block1_conv2_transpose_decoder/kernel/AssignAssign3vgg19_decoder/block1_conv2_transpose_decoder/kernelNvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*&
_output_shapes
:@@
ò
8vgg19_decoder/block1_conv2_transpose_decoder/kernel/readIdentity3vgg19_decoder/block1_conv2_transpose_decoder/kernel*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*&
_output_shapes
:@@
Ö
Cvgg19_decoder/block1_conv2_transpose_decoder/bias/Initializer/zerosConst*
valueB@*    *D
_class:
86loc:@vgg19_decoder/block1_conv2_transpose_decoder/bias*
dtype0*
_output_shapes
:@
¿
1vgg19_decoder/block1_conv2_transpose_decoder/bias
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*D
_class:
86loc:@vgg19_decoder/block1_conv2_transpose_decoder/bias
¥
8vgg19_decoder/block1_conv2_transpose_decoder/bias/AssignAssign1vgg19_decoder/block1_conv2_transpose_decoder/biasCvgg19_decoder/block1_conv2_transpose_decoder/bias/Initializer/zeros*
T0*D
_class:
86loc:@vgg19_decoder/block1_conv2_transpose_decoder/bias*
_output_shapes
:@
à
6vgg19_decoder/block1_conv2_transpose_decoder/bias/readIdentity1vgg19_decoder/block1_conv2_transpose_decoder/bias*
T0*D
_class:
86loc:@vgg19_decoder/block1_conv2_transpose_decoder/bias*
_output_shapes
:@

:vgg19_decoder/block1_conv2_transpose_decoder/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
è
3vgg19_decoder/block1_conv2_transpose_decoder/Conv2DConv2Dvgg19_decoder/Pad_38vgg19_decoder/block1_conv2_transpose_decoder/kernel/read*(
_output_shapes
:@*
T0*
strides
*
paddingVALID
ß
4vgg19_decoder/block1_conv2_transpose_decoder/BiasAddBiasAdd3vgg19_decoder/block1_conv2_transpose_decoder/Conv2D6vgg19_decoder/block1_conv2_transpose_decoder/bias/read*
T0*(
_output_shapes
:@
¢
1vgg19_decoder/block1_conv2_transpose_decoder/ReluRelu4vgg19_decoder/block1_conv2_transpose_decoder/BiasAdd*
T0*(
_output_shapes
:@
á
Jvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @      *<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel
Ë
Hvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/minConst*
valueB
 *ª7¾*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*
dtype0*
_output_shapes
: 
Ë
Hvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/maxConst*
valueB
 *ª7>*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*
dtype0*
_output_shapes
: 
«
Rvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformRandomUniformJvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel
Â
Hvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/subSubHvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/maxHvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel
Ü
Hvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/mulMulRvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/RandomUniformHvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/sub*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*&
_output_shapes
:@*
T0
Î
Dvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniformAddHvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/mulHvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*&
_output_shapes
:@
Ç
)vgg19_decoder/block1_conv1_decoder/kernel
VariableV2*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*
shape:@*
dtype0*&
_output_shapes
:@

0vgg19_decoder/block1_conv1_decoder/kernel/AssignAssign)vgg19_decoder/block1_conv1_decoder/kernelDvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform*&
_output_shapes
:@*
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel
Ô
.vgg19_decoder/block1_conv1_decoder/kernel/readIdentity)vgg19_decoder/block1_conv1_decoder/kernel*&
_output_shapes
:@*
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel
Â
9vgg19_decoder/block1_conv1_decoder/bias/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@vgg19_decoder/block1_conv1_decoder/bias*
dtype0*
_output_shapes
:
«
'vgg19_decoder/block1_conv1_decoder/bias
VariableV2*:
_class0
.,loc:@vgg19_decoder/block1_conv1_decoder/bias*
shape:*
dtype0*
_output_shapes
:
ý
.vgg19_decoder/block1_conv1_decoder/bias/AssignAssign'vgg19_decoder/block1_conv1_decoder/bias9vgg19_decoder/block1_conv1_decoder/bias/Initializer/zeros*:
_class0
.,loc:@vgg19_decoder/block1_conv1_decoder/bias*
_output_shapes
:*
T0
Â
,vgg19_decoder/block1_conv1_decoder/bias/readIdentity'vgg19_decoder/block1_conv1_decoder/bias*
_output_shapes
:*
T0*:
_class0
.,loc:@vgg19_decoder/block1_conv1_decoder/bias

0vgg19_decoder/block1_conv1_decoder/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ò
)vgg19_decoder/block1_conv1_decoder/Conv2DConv2D1vgg19_decoder/block1_conv2_transpose_decoder/Relu.vgg19_decoder/block1_conv1_decoder/kernel/read*
paddingVALID*(
_output_shapes
:*
T0*
strides

Á
*vgg19_decoder/block1_conv1_decoder/BiasAddBiasAdd)vgg19_decoder/block1_conv1_decoder/Conv2D,vgg19_decoder/block1_conv1_decoder/bias/read*
T0*(
_output_shapes
:

'vgg19_decoder/block1_conv1_decoder/TanhTanh*vgg19_decoder/block1_conv1_decoder/BiasAdd*
T0*(
_output_shapes
:
X
vgg19_decoder/mul/xConst*
valueB
 *  C*
dtype0*
_output_shapes
: 

vgg19_decoder/mulMulvgg19_decoder/mul/x'vgg19_decoder/block1_conv1_decoder/Tanh*
T0*(
_output_shapes
:

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_3ce828fa242245009af49faad875f702/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
®0
save/SaveV2/tensor_namesConst"/device:CPU:0*Ò/
valueÈ/BÅ/IB4fast_style_transfer/compressor/compressor/conv1/biasB6fast_style_transfer/compressor/compressor/conv1/kernelB4fast_style_transfer/compressor/compressor/conv2/biasB6fast_style_transfer/compressor/compressor/conv2/kernelB4fast_style_transfer/compressor/compressor/conv3/biasB6fast_style_transfer/compressor/compressor/conv3/kernelB_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/betaB`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gammaBffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_meanBjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_varianceBafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/betaBbfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gammaBhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_meanBlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_varianceBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernelBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernelBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernelBvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/biasBxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernelBvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/biasBxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernelByfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/biasB{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernelB`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/betaBafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gammaBgfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_meanBkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_varianceBbfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/betaBcfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gammaBifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_meanBmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_varianceBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernelBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernelBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernelBxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/biasBzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernelBxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/biasBzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernelB{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/biasB}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernelB8fast_style_transfer/uncompressor/uncompressor/conv1/biasB:fast_style_transfer/uncompressor/uncompressor/conv1/kernelB8fast_style_transfer/uncompressor/uncompressor/conv2/biasB:fast_style_transfer/uncompressor/uncompressor/conv2/kernelB8fast_style_transfer/uncompressor/uncompressor/conv3/biasB:fast_style_transfer/uncompressor/uncompressor/conv3/kernelBglobal_stepB'vgg19_decoder/block1_conv1_decoder/biasB)vgg19_decoder/block1_conv1_decoder/kernelB1vgg19_decoder/block1_conv2_transpose_decoder/biasB3vgg19_decoder/block1_conv2_transpose_decoder/kernelB'vgg19_decoder/block2_conv1_decoder/biasB)vgg19_decoder/block2_conv1_decoder/kernelB1vgg19_decoder/block2_conv2_transpose_decoder/biasB3vgg19_decoder/block2_conv2_transpose_decoder/kernelB'vgg19_decoder/block3_conv1_decoder/biasB)vgg19_decoder/block3_conv1_decoder/kernelBvgg19_encoder/block1_conv1/biasB!vgg19_encoder/block1_conv1/kernelBvgg19_encoder/block1_conv2/biasB!vgg19_encoder/block1_conv2/kernelBvgg19_encoder/block2_conv1/biasB!vgg19_encoder/block2_conv1/kernelBvgg19_encoder/block2_conv2/biasB!vgg19_encoder/block2_conv2/kernelBvgg19_encoder/block3_conv1/biasB!vgg19_encoder/block3_conv1/kernel*
dtype0*
_output_shapes
:I

save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:I*§
valueBIB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
1
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices4fast_style_transfer/compressor/compressor/conv1/bias6fast_style_transfer/compressor/compressor/conv1/kernel4fast_style_transfer/compressor/compressor/conv2/bias6fast_style_transfer/compressor/compressor/conv2/kernel4fast_style_transfer/compressor/compressor/conv3/bias6fast_style_transfer/compressor/compressor/conv3/kernel_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gammaffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_meanjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_varianceafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/betabfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gammahfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_meanlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_varianceufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/biaswfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernelufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/biaswfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernelufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/biaswfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernelvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/biasxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernelvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/biasxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernelyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/betaafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gammagfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_meankfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variancebfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/betacfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gammaifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_meanmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variancewfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/biasyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernelwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/biasyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernelwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/biasyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernelxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/biaszfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernelxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/biaszfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel8fast_style_transfer/uncompressor/uncompressor/conv1/bias:fast_style_transfer/uncompressor/uncompressor/conv1/kernel8fast_style_transfer/uncompressor/uncompressor/conv2/bias:fast_style_transfer/uncompressor/uncompressor/conv2/kernel8fast_style_transfer/uncompressor/uncompressor/conv3/bias:fast_style_transfer/uncompressor/uncompressor/conv3/kernelglobal_step'vgg19_decoder/block1_conv1_decoder/bias)vgg19_decoder/block1_conv1_decoder/kernel1vgg19_decoder/block1_conv2_transpose_decoder/bias3vgg19_decoder/block1_conv2_transpose_decoder/kernel'vgg19_decoder/block2_conv1_decoder/bias)vgg19_decoder/block2_conv1_decoder/kernel1vgg19_decoder/block2_conv2_transpose_decoder/bias3vgg19_decoder/block2_conv2_transpose_decoder/kernel'vgg19_decoder/block3_conv1_decoder/bias)vgg19_decoder/block3_conv1_decoder/kernelvgg19_encoder/block1_conv1/bias!vgg19_encoder/block1_conv1/kernelvgg19_encoder/block1_conv2/bias!vgg19_encoder/block1_conv2/kernelvgg19_encoder/block2_conv1/bias!vgg19_encoder/block2_conv1/kernelvgg19_encoder/block2_conv2/bias!vgg19_encoder/block2_conv2/kernelvgg19_encoder/block3_conv1/bias!vgg19_encoder/block3_conv1/kernel"/device:CPU:0*W
dtypesM
K2I	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
±0
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ò/
valueÈ/BÅ/IB4fast_style_transfer/compressor/compressor/conv1/biasB6fast_style_transfer/compressor/compressor/conv1/kernelB4fast_style_transfer/compressor/compressor/conv2/biasB6fast_style_transfer/compressor/compressor/conv2/kernelB4fast_style_transfer/compressor/compressor/conv3/biasB6fast_style_transfer/compressor/compressor/conv3/kernelB_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/betaB`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gammaBffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_meanBjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_varianceBafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/betaBbfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gammaBhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_meanBlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_varianceBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernelBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernelBufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/biasBwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernelBvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/biasBxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernelBvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/biasBxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernelByfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/biasB{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernelB`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/betaBafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gammaBgfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_meanBkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_varianceBbfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/betaBcfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gammaBifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_meanBmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_varianceBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernelBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernelBwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/biasByfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernelBxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/biasBzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernelBxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/biasBzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernelB{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/biasB}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernelB8fast_style_transfer/uncompressor/uncompressor/conv1/biasB:fast_style_transfer/uncompressor/uncompressor/conv1/kernelB8fast_style_transfer/uncompressor/uncompressor/conv2/biasB:fast_style_transfer/uncompressor/uncompressor/conv2/kernelB8fast_style_transfer/uncompressor/uncompressor/conv3/biasB:fast_style_transfer/uncompressor/uncompressor/conv3/kernelBglobal_stepB'vgg19_decoder/block1_conv1_decoder/biasB)vgg19_decoder/block1_conv1_decoder/kernelB1vgg19_decoder/block1_conv2_transpose_decoder/biasB3vgg19_decoder/block1_conv2_transpose_decoder/kernelB'vgg19_decoder/block2_conv1_decoder/biasB)vgg19_decoder/block2_conv1_decoder/kernelB1vgg19_decoder/block2_conv2_transpose_decoder/biasB3vgg19_decoder/block2_conv2_transpose_decoder/kernelB'vgg19_decoder/block3_conv1_decoder/biasB)vgg19_decoder/block3_conv1_decoder/kernelBvgg19_encoder/block1_conv1/biasB!vgg19_encoder/block1_conv1/kernelBvgg19_encoder/block1_conv2/biasB!vgg19_encoder/block1_conv2/kernelBvgg19_encoder/block2_conv1/biasB!vgg19_encoder/block2_conv1/kernelBvgg19_encoder/block2_conv2/biasB!vgg19_encoder/block2_conv2/kernelBvgg19_encoder/block3_conv1/biasB!vgg19_encoder/block3_conv1/kernel*
dtype0*
_output_shapes
:I

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*§
valueBIB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:I

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	
Ê
save/AssignAssign4fast_style_transfer/compressor/compressor/conv1/biassave/RestoreV2*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv1/bias*
_output_shapes	
:
ß
save/Assign_1Assign6fast_style_transfer/compressor/compressor/conv1/kernelsave/RestoreV2:1*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv1/kernel*(
_output_shapes
:
Î
save/Assign_2Assign4fast_style_transfer/compressor/compressor/conv2/biassave/RestoreV2:2*
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv2/bias*
_output_shapes	
:
ß
save/Assign_3Assign6fast_style_transfer/compressor/compressor/conv2/kernelsave/RestoreV2:3*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv2/kernel*(
_output_shapes
:
Í
save/Assign_4Assign4fast_style_transfer/compressor/compressor/conv3/biassave/RestoreV2:4*
_output_shapes
: *
T0*G
_class=
;9loc:@fast_style_transfer/compressor/compressor/conv3/bias
Þ
save/Assign_5Assign6fast_style_transfer/compressor/compressor/conv3/kernelsave/RestoreV2:5*
T0*I
_class?
=;loc:@fast_style_transfer/compressor/compressor/conv3/kernel*'
_output_shapes
: 
¤
save/Assign_6Assign_fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/betasave/RestoreV2:6*
_output_shapes	
:*
T0*r
_classh
fdloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta
¦
save/Assign_7Assign`fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gammasave/RestoreV2:7*
T0*s
_classi
geloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma*
_output_shapes	
:
²
save/Assign_8Assignffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_meansave/RestoreV2:8*
T0*y
_classo
mkloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean*
_output_shapes	
:
º
save/Assign_9Assignjfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variancesave/RestoreV2:9*
T0*}
_classs
qoloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance*
_output_shapes	
:
ª
save/Assign_10Assignafast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/betasave/RestoreV2:10*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta*
_output_shapes	
:*
T0
¬
save/Assign_11Assignbfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gammasave/RestoreV2:11*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma*
_output_shapes	
:
¸
save/Assign_12Assignhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_meansave/RestoreV2:12*
T0*{
_classq
omloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:
À
save/Assign_13Assignlfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variancesave/RestoreV2:13*
_output_shapes	
:*
T0*
_classu
sqloc:@fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance
Ó
save/Assign_14Assignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/biassave/RestoreV2:14*
_output_shapes	
:*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias
å
save/Assign_15Assignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernelsave/RestoreV2:15*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel*(
_output_shapes
:
Ò
save/Assign_16Assignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/biassave/RestoreV2:16*
_output_shapes
:@*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias
ä
save/Assign_17Assignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernelsave/RestoreV2:17*'
_output_shapes
:@*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel
Ò
save/Assign_18Assignufast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/biassave/RestoreV2:18*
T0*
_class~
|zloc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias*
_output_shapes
: 
ã
save/Assign_19Assignwfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernelsave/RestoreV2:19*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel*&
_output_shapes
:@ 
Õ
save/Assign_20Assignvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/biassave/RestoreV2:20*
_output_shapes	
:*
T0*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias
Þ
save/Assign_21Assignxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernelsave/RestoreV2:21*
_output_shapes
:	 *
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel
Õ
save/Assign_22Assignvfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/biassave/RestoreV2:22*
_class
}{loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias*
_output_shapes	
:*
T0
ß
save/Assign_23Assignxfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernelsave/RestoreV2:23*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel* 
_output_shapes
:

Ü
save/Assign_24Assignyfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/biassave/RestoreV2:24*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias*
_output_shapes
: 
æ
save/Assign_25Assign{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernelsave/RestoreV2:25*
_output_shapes
:	 *
T0*
_class
loc:@fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel
¨
save/Assign_26Assign`fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/betasave/RestoreV2:26*
T0*s
_classi
geloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta*
_output_shapes	
:
ª
save/Assign_27Assignafast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gammasave/RestoreV2:27*
T0*t
_classj
hfloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma*
_output_shapes	
:
¶
save/Assign_28Assigngfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_meansave/RestoreV2:28*z
_classp
nlloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean*
_output_shapes	
:*
T0
¾
save/Assign_29Assignkfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variancesave/RestoreV2:29*
T0*~
_classt
rploc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance*
_output_shapes	
:
¬
save/Assign_30Assignbfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/betasave/RestoreV2:30*
T0*u
_classk
igloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta*
_output_shapes	
:
®
save/Assign_31Assigncfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gammasave/RestoreV2:31*
T0*v
_classl
jhloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma*
_output_shapes	
:
º
save/Assign_32Assignifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_meansave/RestoreV2:32*
T0*|
_classr
pnloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean*
_output_shapes	
:
Ã
save/Assign_33Assignmfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variancesave/RestoreV2:33*
_output_shapes	
:*
T0*
_classv
trloc:@fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance
Ø
save/Assign_34Assignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/biassave/RestoreV2:34*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias*
_output_shapes	
:
ê
save/Assign_35Assignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernelsave/RestoreV2:35*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel*(
_output_shapes
:
×
save/Assign_36Assignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/biassave/RestoreV2:36*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias*
_output_shapes
:@
é
save/Assign_37Assignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernelsave/RestoreV2:37*
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel*'
_output_shapes
:@
×
save/Assign_38Assignwfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/biassave/RestoreV2:38*
T0*
_class
~|loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias*
_output_shapes
: 
è
save/Assign_39Assignyfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernelsave/RestoreV2:39*&
_output_shapes
:@ *
T0*
_class
~loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel
Ú
save/Assign_40Assignxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/biassave/RestoreV2:40*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias*
_output_shapes	
:*
T0
ã
save/Assign_41Assignzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernelsave/RestoreV2:41*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel*
_output_shapes
:	 *
T0
Ú
save/Assign_42Assignxfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/biassave/RestoreV2:42*
T0*
_class
}loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias*
_output_shapes	
:
ä
save/Assign_43Assignzfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernelsave/RestoreV2:43*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel* 
_output_shapes
:

á
save/Assign_44Assign{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/biassave/RestoreV2:44*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias*
_output_shapes
: 
ê
save/Assign_45Assign}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernelsave/RestoreV2:45*
T0*
_class
loc:@fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel*
_output_shapes
:	 
×
save/Assign_46Assign8fast_style_transfer/uncompressor/uncompressor/conv1/biassave/RestoreV2:46*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv1/bias*
_output_shapes
:@
ç
save/Assign_47Assign:fast_style_transfer/uncompressor/uncompressor/conv1/kernelsave/RestoreV2:47*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv1/kernel*&
_output_shapes
: @
Ø
save/Assign_48Assign8fast_style_transfer/uncompressor/uncompressor/conv2/biassave/RestoreV2:48*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv2/bias*
_output_shapes	
:
è
save/Assign_49Assign:fast_style_transfer/uncompressor/uncompressor/conv2/kernelsave/RestoreV2:49*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv2/kernel*'
_output_shapes
:@
Ø
save/Assign_50Assign8fast_style_transfer/uncompressor/uncompressor/conv3/biassave/RestoreV2:50*
T0*K
_classA
?=loc:@fast_style_transfer/uncompressor/uncompressor/conv3/bias*
_output_shapes	
:
é
save/Assign_51Assign:fast_style_transfer/uncompressor/uncompressor/conv3/kernelsave/RestoreV2:51*
T0*M
_classC
A?loc:@fast_style_transfer/uncompressor/uncompressor/conv3/kernel*(
_output_shapes
:
y
save/Assign_52Assignglobal_stepsave/RestoreV2:52*
T0	*
_class
loc:@global_step*
_output_shapes
: 
µ
save/Assign_53Assign'vgg19_decoder/block1_conv1_decoder/biassave/RestoreV2:53*
_output_shapes
:*
T0*:
_class0
.,loc:@vgg19_decoder/block1_conv1_decoder/bias
Å
save/Assign_54Assign)vgg19_decoder/block1_conv1_decoder/kernelsave/RestoreV2:54*
T0*<
_class2
0.loc:@vgg19_decoder/block1_conv1_decoder/kernel*&
_output_shapes
:@
É
save/Assign_55Assign1vgg19_decoder/block1_conv2_transpose_decoder/biassave/RestoreV2:55*
T0*D
_class:
86loc:@vgg19_decoder/block1_conv2_transpose_decoder/bias*
_output_shapes
:@
Ù
save/Assign_56Assign3vgg19_decoder/block1_conv2_transpose_decoder/kernelsave/RestoreV2:56*
T0*F
_class<
:8loc:@vgg19_decoder/block1_conv2_transpose_decoder/kernel*&
_output_shapes
:@@
µ
save/Assign_57Assign'vgg19_decoder/block2_conv1_decoder/biassave/RestoreV2:57*
_output_shapes
:@*
T0*:
_class0
.,loc:@vgg19_decoder/block2_conv1_decoder/bias
Æ
save/Assign_58Assign)vgg19_decoder/block2_conv1_decoder/kernelsave/RestoreV2:58*
T0*<
_class2
0.loc:@vgg19_decoder/block2_conv1_decoder/kernel*'
_output_shapes
:@
Ê
save/Assign_59Assign1vgg19_decoder/block2_conv2_transpose_decoder/biassave/RestoreV2:59*
T0*D
_class:
86loc:@vgg19_decoder/block2_conv2_transpose_decoder/bias*
_output_shapes	
:
Û
save/Assign_60Assign3vgg19_decoder/block2_conv2_transpose_decoder/kernelsave/RestoreV2:60*
T0*F
_class<
:8loc:@vgg19_decoder/block2_conv2_transpose_decoder/kernel*(
_output_shapes
:
¶
save/Assign_61Assign'vgg19_decoder/block3_conv1_decoder/biassave/RestoreV2:61*
T0*:
_class0
.,loc:@vgg19_decoder/block3_conv1_decoder/bias*
_output_shapes	
:
Ç
save/Assign_62Assign)vgg19_decoder/block3_conv1_decoder/kernelsave/RestoreV2:62*
T0*<
_class2
0.loc:@vgg19_decoder/block3_conv1_decoder/kernel*(
_output_shapes
:
¥
save/Assign_63Assignvgg19_encoder/block1_conv1/biassave/RestoreV2:63*
_output_shapes
:@*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv1/bias
µ
save/Assign_64Assign!vgg19_encoder/block1_conv1/kernelsave/RestoreV2:64*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv1/kernel*&
_output_shapes
:@
¥
save/Assign_65Assignvgg19_encoder/block1_conv2/biassave/RestoreV2:65*
_output_shapes
:@*
T0*2
_class(
&$loc:@vgg19_encoder/block1_conv2/bias
µ
save/Assign_66Assign!vgg19_encoder/block1_conv2/kernelsave/RestoreV2:66*&
_output_shapes
:@@*
T0*4
_class*
(&loc:@vgg19_encoder/block1_conv2/kernel
¦
save/Assign_67Assignvgg19_encoder/block2_conv1/biassave/RestoreV2:67*2
_class(
&$loc:@vgg19_encoder/block2_conv1/bias*
_output_shapes	
:*
T0
¶
save/Assign_68Assign!vgg19_encoder/block2_conv1/kernelsave/RestoreV2:68*4
_class*
(&loc:@vgg19_encoder/block2_conv1/kernel*'
_output_shapes
:@*
T0
¦
save/Assign_69Assignvgg19_encoder/block2_conv2/biassave/RestoreV2:69*
T0*2
_class(
&$loc:@vgg19_encoder/block2_conv2/bias*
_output_shapes	
:
·
save/Assign_70Assign!vgg19_encoder/block2_conv2/kernelsave/RestoreV2:70*
T0*4
_class*
(&loc:@vgg19_encoder/block2_conv2/kernel*(
_output_shapes
:
¦
save/Assign_71Assignvgg19_encoder/block3_conv1/biassave/RestoreV2:71*
T0*2
_class(
&$loc:@vgg19_encoder/block3_conv1/bias*
_output_shapes	
:
·
save/Assign_72Assign!vgg19_encoder/block3_conv1/kernelsave/RestoreV2:72*(
_output_shapes
:*
T0*4
_class*
(&loc:@vgg19_encoder/block3_conv1/kernel
ç	
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
trainable_variables~~

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Initializer/zeros:08

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Initializer/zeros:08

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Initializer/zeros:08

zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform:08

xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias:0}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Assign}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Initializer/zeros:08

zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform:08

xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias:0}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Assign}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Initializer/zeros:08
¦
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform:08

{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Initializer/zeros:08
¢
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform:08

zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Initializer/zeros:08
¢
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform:08

zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Initializer/zeros:08
®
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform:08

}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv1/kernel:0=fast_style_transfer/compressor/compressor/conv1/kernel/Assign=fast_style_transfer/compressor/compressor/conv1/kernel/read:02Sfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv1/bias:0;fast_style_transfer/compressor/compressor/conv1/bias/Assign;fast_style_transfer/compressor/compressor/conv1/bias/read:02Hfast_style_transfer/compressor/compressor/conv1/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv2/kernel:0=fast_style_transfer/compressor/compressor/conv2/kernel/Assign=fast_style_transfer/compressor/compressor/conv2/kernel/read:02Sfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv2/bias:0;fast_style_transfer/compressor/compressor/conv2/bias/Assign;fast_style_transfer/compressor/compressor/conv2/bias/read:02Hfast_style_transfer/compressor/compressor/conv2/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv3/kernel:0=fast_style_transfer/compressor/compressor/conv3/kernel/Assign=fast_style_transfer/compressor/compressor/conv3/kernel/read:02Sfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv3/bias:0;fast_style_transfer/compressor/compressor/conv3/bias/Assign;fast_style_transfer/compressor/compressor/conv3/bias/read:02Hfast_style_transfer/compressor/compressor/conv3/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv1/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv1/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv1/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv1/bias:0?fast_style_transfer/uncompressor/uncompressor/conv1/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv1/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv1/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv2/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv2/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv2/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv2/bias:0?fast_style_transfer/uncompressor/uncompressor/conv2/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv2/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv2/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv3/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv3/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv3/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv3/bias:0?fast_style_transfer/uncompressor/uncompressor/conv3/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv3/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv3/bias/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"×
	variablesýÖùÖ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
¹
#vgg19_encoder/block1_conv1/kernel:0(vgg19_encoder/block1_conv1/kernel/Assign(vgg19_encoder/block1_conv1/kernel/read:02>vgg19_encoder/block1_conv1/kernel/Initializer/random_uniform:0
¨
!vgg19_encoder/block1_conv1/bias:0&vgg19_encoder/block1_conv1/bias/Assign&vgg19_encoder/block1_conv1/bias/read:023vgg19_encoder/block1_conv1/bias/Initializer/zeros:0
¹
#vgg19_encoder/block1_conv2/kernel:0(vgg19_encoder/block1_conv2/kernel/Assign(vgg19_encoder/block1_conv2/kernel/read:02>vgg19_encoder/block1_conv2/kernel/Initializer/random_uniform:0
¨
!vgg19_encoder/block1_conv2/bias:0&vgg19_encoder/block1_conv2/bias/Assign&vgg19_encoder/block1_conv2/bias/read:023vgg19_encoder/block1_conv2/bias/Initializer/zeros:0
¹
#vgg19_encoder/block2_conv1/kernel:0(vgg19_encoder/block2_conv1/kernel/Assign(vgg19_encoder/block2_conv1/kernel/read:02>vgg19_encoder/block2_conv1/kernel/Initializer/random_uniform:0
¨
!vgg19_encoder/block2_conv1/bias:0&vgg19_encoder/block2_conv1/bias/Assign&vgg19_encoder/block2_conv1/bias/read:023vgg19_encoder/block2_conv1/bias/Initializer/zeros:0
¹
#vgg19_encoder/block2_conv2/kernel:0(vgg19_encoder/block2_conv2/kernel/Assign(vgg19_encoder/block2_conv2/kernel/read:02>vgg19_encoder/block2_conv2/kernel/Initializer/random_uniform:0
¨
!vgg19_encoder/block2_conv2/bias:0&vgg19_encoder/block2_conv2/bias/Assign&vgg19_encoder/block2_conv2/bias/read:023vgg19_encoder/block2_conv2/bias/Initializer/zeros:0
¹
#vgg19_encoder/block3_conv1/kernel:0(vgg19_encoder/block3_conv1/kernel/Assign(vgg19_encoder/block3_conv1/kernel/read:02>vgg19_encoder/block3_conv1/kernel/Initializer/random_uniform:0
¨
!vgg19_encoder/block3_conv1/bias:0&vgg19_encoder/block3_conv1/bias/Assign&vgg19_encoder/block3_conv1/bias/read:023vgg19_encoder/block3_conv1/bias/Initializer/zeros:0

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv1/bias/Initializer/zeros:08

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv2/bias/Initializer/zeros:08

yfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel:0~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Assign~fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/kernel/Initializer/random_uniform:08

wfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias:0|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Assign|fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/conv3/bias/Initializer/zeros:08

zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/kernel/Initializer/random_uniform:08

xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias:0}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Assign}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense1/bias/Initializer/zeros:08
«
bfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma:0gfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/Assigngfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/read:02sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/gamma/Initializer/ones:0
¨
afast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta:0ffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/Assignffast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/read:02sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/beta/Initializer/zeros:0
Ä
hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean:0mfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/Assignmfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/read:02zfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_mean/Initializer/zeros:0
Ó
lfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance:0qfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/Assignqfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/read:02}fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization/moving_variance/Initializer/ones:0

zfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/kernel/Initializer/random_uniform:08

xfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias:0}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Assign}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense2/bias/Initializer/zeros:08
³
dfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma:0ifast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/Assignifast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/read:02ufast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/gamma/Initializer/ones:0
°
cfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta:0hfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/Assignhfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/read:02ufast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/beta/Initializer/zeros:0
Ì
jfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean:0ofast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/Assignofast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/read:02|fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_mean/Initializer/zeros:0
Û
nfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance:0sfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/Assignsfast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/read:02fast_style_transfer/transformation/transformation/base_image_transform/batch_normalization_1/moving_variance/Initializer/ones:0
¦
}fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/kernel/Initializer/random_uniform:08

{fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias:0fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Assignfast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/read:02fast_style_transfer/transformation/transformation/base_image_transform/transformation/base_image_transform/dense_out/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv1/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv2/bias/Initializer/zeros:08

{fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/kernel/Initializer/random_uniform:08

yfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias:0~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Assign~fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/conv3/bias/Initializer/zeros:08
¢
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/kernel/Initializer/random_uniform:08

zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense1/bias/Initializer/zeros:08
¯
cfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma:0hfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/Assignhfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/read:02tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/gamma/Initializer/ones:0
¬
bfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta:0gfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/Assigngfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/read:02tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/beta/Initializer/zeros:0
È
ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean:0nfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/Assignnfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/read:02{fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_mean/Initializer/zeros:0
×
mfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance:0rfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/Assignrfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/read:02~fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization/moving_variance/Initializer/ones:0
¢
|fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/kernel/Initializer/random_uniform:08

zfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense2/bias/Initializer/zeros:08
·
efast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma:0jfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/Assignjfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/read:02vfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/gamma/Initializer/ones:0
´
dfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta:0ifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/Assignifast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/read:02vfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/beta/Initializer/zeros:0
Ð
kfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean:0pfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/Assignpfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/read:02}fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_mean/Initializer/zeros:0
à
ofast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance:0tfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/Assigntfast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/read:02fast_style_transfer/transformation/transformation/style_image_transform/batch_normalization_1/moving_variance/Initializer/ones:0
®
fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/kernel/Initializer/random_uniform:08

}fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias:0fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Assignfast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/read:02fast_style_transfer/transformation/transformation/style_image_transform/transformation/style_image_transform/dense_out/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv1/kernel:0=fast_style_transfer/compressor/compressor/conv1/kernel/Assign=fast_style_transfer/compressor/compressor/conv1/kernel/read:02Sfast_style_transfer/compressor/compressor/conv1/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv1/bias:0;fast_style_transfer/compressor/compressor/conv1/bias/Assign;fast_style_transfer/compressor/compressor/conv1/bias/read:02Hfast_style_transfer/compressor/compressor/conv1/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv2/kernel:0=fast_style_transfer/compressor/compressor/conv2/kernel/Assign=fast_style_transfer/compressor/compressor/conv2/kernel/read:02Sfast_style_transfer/compressor/compressor/conv2/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv2/bias:0;fast_style_transfer/compressor/compressor/conv2/bias/Assign;fast_style_transfer/compressor/compressor/conv2/bias/read:02Hfast_style_transfer/compressor/compressor/conv2/bias/Initializer/zeros:08

8fast_style_transfer/compressor/compressor/conv3/kernel:0=fast_style_transfer/compressor/compressor/conv3/kernel/Assign=fast_style_transfer/compressor/compressor/conv3/kernel/read:02Sfast_style_transfer/compressor/compressor/conv3/kernel/Initializer/random_uniform:08
þ
6fast_style_transfer/compressor/compressor/conv3/bias:0;fast_style_transfer/compressor/compressor/conv3/bias/Assign;fast_style_transfer/compressor/compressor/conv3/bias/read:02Hfast_style_transfer/compressor/compressor/conv3/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv1/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv1/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv1/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv1/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv1/bias:0?fast_style_transfer/uncompressor/uncompressor/conv1/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv1/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv1/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv2/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv2/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv2/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv2/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv2/bias:0?fast_style_transfer/uncompressor/uncompressor/conv2/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv2/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv2/bias/Initializer/zeros:08

<fast_style_transfer/uncompressor/uncompressor/conv3/kernel:0Afast_style_transfer/uncompressor/uncompressor/conv3/kernel/AssignAfast_style_transfer/uncompressor/uncompressor/conv3/kernel/read:02Wfast_style_transfer/uncompressor/uncompressor/conv3/kernel/Initializer/random_uniform:08

:fast_style_transfer/uncompressor/uncompressor/conv3/bias:0?fast_style_transfer/uncompressor/uncompressor/conv3/bias/Assign?fast_style_transfer/uncompressor/uncompressor/conv3/bias/read:02Lfast_style_transfer/uncompressor/uncompressor/conv3/bias/Initializer/zeros:08
Ù
+vgg19_decoder/block3_conv1_decoder/kernel:00vgg19_decoder/block3_conv1_decoder/kernel/Assign0vgg19_decoder/block3_conv1_decoder/kernel/read:02Fvgg19_decoder/block3_conv1_decoder/kernel/Initializer/random_uniform:0
È
)vgg19_decoder/block3_conv1_decoder/bias:0.vgg19_decoder/block3_conv1_decoder/bias/Assign.vgg19_decoder/block3_conv1_decoder/bias/read:02;vgg19_decoder/block3_conv1_decoder/bias/Initializer/zeros:0

5vgg19_decoder/block2_conv2_transpose_decoder/kernel:0:vgg19_decoder/block2_conv2_transpose_decoder/kernel/Assign:vgg19_decoder/block2_conv2_transpose_decoder/kernel/read:02Pvgg19_decoder/block2_conv2_transpose_decoder/kernel/Initializer/random_uniform:0
ð
3vgg19_decoder/block2_conv2_transpose_decoder/bias:08vgg19_decoder/block2_conv2_transpose_decoder/bias/Assign8vgg19_decoder/block2_conv2_transpose_decoder/bias/read:02Evgg19_decoder/block2_conv2_transpose_decoder/bias/Initializer/zeros:0
Ù
+vgg19_decoder/block2_conv1_decoder/kernel:00vgg19_decoder/block2_conv1_decoder/kernel/Assign0vgg19_decoder/block2_conv1_decoder/kernel/read:02Fvgg19_decoder/block2_conv1_decoder/kernel/Initializer/random_uniform:0
È
)vgg19_decoder/block2_conv1_decoder/bias:0.vgg19_decoder/block2_conv1_decoder/bias/Assign.vgg19_decoder/block2_conv1_decoder/bias/read:02;vgg19_decoder/block2_conv1_decoder/bias/Initializer/zeros:0

5vgg19_decoder/block1_conv2_transpose_decoder/kernel:0:vgg19_decoder/block1_conv2_transpose_decoder/kernel/Assign:vgg19_decoder/block1_conv2_transpose_decoder/kernel/read:02Pvgg19_decoder/block1_conv2_transpose_decoder/kernel/Initializer/random_uniform:0
ð
3vgg19_decoder/block1_conv2_transpose_decoder/bias:08vgg19_decoder/block1_conv2_transpose_decoder/bias/Assign8vgg19_decoder/block1_conv2_transpose_decoder/bias/read:02Evgg19_decoder/block1_conv2_transpose_decoder/bias/Initializer/zeros:0
Ù
+vgg19_decoder/block1_conv1_decoder/kernel:00vgg19_decoder/block1_conv1_decoder/kernel/Assign0vgg19_decoder/block1_conv1_decoder/kernel/read:02Fvgg19_decoder/block1_conv1_decoder/kernel/Initializer/random_uniform:0
È
)vgg19_decoder/block1_conv1_decoder/bias:0.vgg19_decoder/block1_conv1_decoder/bias/Assign.vgg19_decoder/block1_conv1_decoder/bias/read:02;vgg19_decoder/block1_conv1_decoder/bias/Initializer/zeros:0*
serving_default÷
(
examples
input_example_tensor:0 6
content_images$
ExpandDims:0?
generated_images+
vgg19_decoder/mul:06
style_images&
ExpandDims_1:0tensorflow/serving/predict
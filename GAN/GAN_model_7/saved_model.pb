??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d??* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
d??*
dtype0
t
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_namedense_14/bias
m
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes

:??*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_29/kernel
~
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_6/kernel
?
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
: *
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ?*!
shared_nameconv2d_28/kernel
~
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*'
_output_shapes
: ?*
dtype0
u
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_28/bias
n
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes	
:?*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_27/kernel
~
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*'
_output_shapes
:?*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?$
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
R
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
R
9	variables
:regularization_losses
;trainable_variables
<	keras_api
h

=kernel
>bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
F
0
1
2
 3
)4
*5
36
47
=8
>9
 
F
0
1
2
 3
)4
*5
36
47
=8
>9
?
Cnon_trainable_variables
Dlayer_metrics

Elayers
	variables
regularization_losses
Flayer_regularization_losses
trainable_variables
Gmetrics
 
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Hnon_trainable_variables
Ilayer_metrics

Jlayers
	variables
regularization_losses
Klayer_regularization_losses
trainable_variables
Lmetrics
 
 
 
?
Mnon_trainable_variables
Nlayer_metrics

Olayers
	variables
regularization_losses
Player_regularization_losses
trainable_variables
Qmetrics
 
 
 
?
Rnon_trainable_variables
Slayer_metrics

Tlayers
	variables
regularization_losses
Ulayer_regularization_losses
trainable_variables
Vmetrics
\Z
VARIABLE_VALUEconv2d_29/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_29/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?
Wnon_trainable_variables
Xlayer_metrics

Ylayers
!	variables
"regularization_losses
Zlayer_regularization_losses
#trainable_variables
[metrics
 
 
 
?
\non_trainable_variables
]layer_metrics

^layers
%	variables
&regularization_losses
_layer_regularization_losses
'trainable_variables
`metrics
ec
VARIABLE_VALUEconv2d_transpose_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?
anon_trainable_variables
blayer_metrics

clayers
+	variables
,regularization_losses
dlayer_regularization_losses
-trainable_variables
emetrics
 
 
 
?
fnon_trainable_variables
glayer_metrics

hlayers
/	variables
0regularization_losses
ilayer_regularization_losses
1trainable_variables
jmetrics
\Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_28/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?
knon_trainable_variables
llayer_metrics

mlayers
5	variables
6regularization_losses
nlayer_regularization_losses
7trainable_variables
ometrics
 
 
 
?
pnon_trainable_variables
qlayer_metrics

rlayers
9	variables
:regularization_losses
slayer_regularization_losses
;trainable_variables
tmetrics
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
?
unon_trainable_variables
vlayer_metrics

wlayers
?	variables
@regularization_losses
xlayer_regularization_losses
Atrainable_variables
ymetrics
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
{
serving_default_input_19Placeholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19dense_14/kerneldense_14/biasconv2d_29/kernelconv2d_29/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_28/kernelconv2d_28/biasconv2d_27/kernelconv2d_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *,
f'R%
#__inference_signature_wrapper_95577
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *'
f"R 
__inference__traced_save_95940
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/biasconv2d_29/kernelconv2d_29/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_28/kernelconv2d_28/biasconv2d_27/kernelconv2d_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__traced_restore_95980??
?e
?
 __inference__wrapped_model_95164
input_194
0model_18_dense_14_matmul_readvariableop_resource5
1model_18_dense_14_biasadd_readvariableop_resource5
1model_18_conv2d_29_conv2d_readvariableop_resource6
2model_18_conv2d_29_biasadd_readvariableop_resourceH
Dmodel_18_conv2d_transpose_6_conv2d_transpose_readvariableop_resource?
;model_18_conv2d_transpose_6_biasadd_readvariableop_resource5
1model_18_conv2d_28_conv2d_readvariableop_resource6
2model_18_conv2d_28_biasadd_readvariableop_resource5
1model_18_conv2d_27_conv2d_readvariableop_resource6
2model_18_conv2d_27_biasadd_readvariableop_resource
identity??)model_18/conv2d_27/BiasAdd/ReadVariableOp?(model_18/conv2d_27/Conv2D/ReadVariableOp?)model_18/conv2d_28/BiasAdd/ReadVariableOp?(model_18/conv2d_28/Conv2D/ReadVariableOp?)model_18/conv2d_29/BiasAdd/ReadVariableOp?(model_18/conv2d_29/Conv2D/ReadVariableOp?2model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp?;model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?(model_18/dense_14/BiasAdd/ReadVariableOp?'model_18/dense_14/MatMul/ReadVariableOp?
'model_18/dense_14/MatMul/ReadVariableOpReadVariableOp0model_18_dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02)
'model_18/dense_14/MatMul/ReadVariableOp?
model_18/dense_14/MatMulMatMulinput_19/model_18/dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
model_18/dense_14/MatMul?
(model_18/dense_14/BiasAdd/ReadVariableOpReadVariableOp1model_18_dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02*
(model_18/dense_14/BiasAdd/ReadVariableOp?
model_18/dense_14/BiasAddBiasAdd"model_18/dense_14/MatMul:product:00model_18/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
model_18/dense_14/BiasAdd?
!model_18/leaky_re_lu_33/LeakyRelu	LeakyRelu"model_18/dense_14/BiasAdd:output:0*)
_output_shapes
:???????????*
alpha%???>2#
!model_18/leaky_re_lu_33/LeakyRelu?
model_18/reshape_6/ShapeShape/model_18/leaky_re_lu_33/LeakyRelu:activations:0*
T0*
_output_shapes
:2
model_18/reshape_6/Shape?
&model_18/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_18/reshape_6/strided_slice/stack?
(model_18/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_18/reshape_6/strided_slice/stack_1?
(model_18/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_18/reshape_6/strided_slice/stack_2?
 model_18/reshape_6/strided_sliceStridedSlice!model_18/reshape_6/Shape:output:0/model_18/reshape_6/strided_slice/stack:output:01model_18/reshape_6/strided_slice/stack_1:output:01model_18/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_18/reshape_6/strided_slice?
"model_18/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_18/reshape_6/Reshape/shape/1?
"model_18/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_18/reshape_6/Reshape/shape/2?
"model_18/reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2$
"model_18/reshape_6/Reshape/shape/3?
 model_18/reshape_6/Reshape/shapePack)model_18/reshape_6/strided_slice:output:0+model_18/reshape_6/Reshape/shape/1:output:0+model_18/reshape_6/Reshape/shape/2:output:0+model_18/reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_18/reshape_6/Reshape/shape?
model_18/reshape_6/ReshapeReshape/model_18/leaky_re_lu_33/LeakyRelu:activations:0)model_18/reshape_6/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
model_18/reshape_6/Reshape?
(model_18/conv2d_29/Conv2D/ReadVariableOpReadVariableOp1model_18_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02*
(model_18/conv2d_29/Conv2D/ReadVariableOp?
model_18/conv2d_29/Conv2DConv2D#model_18/reshape_6/Reshape:output:00model_18/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_18/conv2d_29/Conv2D?
)model_18/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp2model_18_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_18/conv2d_29/BiasAdd/ReadVariableOp?
model_18/conv2d_29/BiasAddBiasAdd"model_18/conv2d_29/Conv2D:output:01model_18/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_18/conv2d_29/BiasAdd?
!model_18/leaky_re_lu_32/LeakyRelu	LeakyRelu#model_18/conv2d_29/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2#
!model_18/leaky_re_lu_32/LeakyRelu?
!model_18/conv2d_transpose_6/ShapeShape/model_18/leaky_re_lu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:2#
!model_18/conv2d_transpose_6/Shape?
/model_18/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_18/conv2d_transpose_6/strided_slice/stack?
1model_18/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_18/conv2d_transpose_6/strided_slice/stack_1?
1model_18/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_18/conv2d_transpose_6/strided_slice/stack_2?
)model_18/conv2d_transpose_6/strided_sliceStridedSlice*model_18/conv2d_transpose_6/Shape:output:08model_18/conv2d_transpose_6/strided_slice/stack:output:0:model_18/conv2d_transpose_6/strided_slice/stack_1:output:0:model_18/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_18/conv2d_transpose_6/strided_slice?
#model_18/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_18/conv2d_transpose_6/stack/1?
#model_18/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_18/conv2d_transpose_6/stack/2?
#model_18/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_18/conv2d_transpose_6/stack/3?
!model_18/conv2d_transpose_6/stackPack2model_18/conv2d_transpose_6/strided_slice:output:0,model_18/conv2d_transpose_6/stack/1:output:0,model_18/conv2d_transpose_6/stack/2:output:0,model_18/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!model_18/conv2d_transpose_6/stack?
1model_18/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1model_18/conv2d_transpose_6/strided_slice_1/stack?
3model_18/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3model_18/conv2d_transpose_6/strided_slice_1/stack_1?
3model_18/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3model_18/conv2d_transpose_6/strided_slice_1/stack_2?
+model_18/conv2d_transpose_6/strided_slice_1StridedSlice*model_18/conv2d_transpose_6/stack:output:0:model_18/conv2d_transpose_6/strided_slice_1/stack:output:0<model_18/conv2d_transpose_6/strided_slice_1/stack_1:output:0<model_18/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+model_18/conv2d_transpose_6/strided_slice_1?
;model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_18_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02=
;model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
,model_18/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput*model_18/conv2d_transpose_6/stack:output:0Cmodel_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0/model_18/leaky_re_lu_32/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,model_18/conv2d_transpose_6/conv2d_transpose?
2model_18/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp;model_18_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp?
#model_18/conv2d_transpose_6/BiasAddBiasAdd5model_18/conv2d_transpose_6/conv2d_transpose:output:0:model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#model_18/conv2d_transpose_6/BiasAdd?
!model_18/leaky_re_lu_31/LeakyRelu	LeakyRelu,model_18/conv2d_transpose_6/BiasAdd:output:0*/
_output_shapes
:?????????  *
alpha%???>2#
!model_18/leaky_re_lu_31/LeakyRelu?
(model_18/conv2d_28/Conv2D/ReadVariableOpReadVariableOp1model_18_conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype02*
(model_18/conv2d_28/Conv2D/ReadVariableOp?
model_18/conv2d_28/Conv2DConv2D/model_18/leaky_re_lu_31/LeakyRelu:activations:00model_18/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
model_18/conv2d_28/Conv2D?
)model_18/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp2model_18_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)model_18/conv2d_28/BiasAdd/ReadVariableOp?
model_18/conv2d_28/BiasAddBiasAdd"model_18/conv2d_28/Conv2D:output:01model_18/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
model_18/conv2d_28/BiasAdd?
!model_18/leaky_re_lu_30/LeakyRelu	LeakyRelu#model_18/conv2d_28/BiasAdd:output:0*0
_output_shapes
:????????? ?*
alpha%???>2#
!model_18/leaky_re_lu_30/LeakyRelu?
(model_18/conv2d_27/Conv2D/ReadVariableOpReadVariableOp1model_18_conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02*
(model_18/conv2d_27/Conv2D/ReadVariableOp?
model_18/conv2d_27/Conv2DConv2D/model_18/leaky_re_lu_30/LeakyRelu:activations:00model_18/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model_18/conv2d_27/Conv2D?
)model_18/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp2model_18_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_18/conv2d_27/BiasAdd/ReadVariableOp?
model_18/conv2d_27/BiasAddBiasAdd"model_18/conv2d_27/Conv2D:output:01model_18/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model_18/conv2d_27/BiasAdd?
model_18/conv2d_27/TanhTanh#model_18/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model_18/conv2d_27/Tanh?
IdentityIdentitymodel_18/conv2d_27/Tanh:y:0*^model_18/conv2d_27/BiasAdd/ReadVariableOp)^model_18/conv2d_27/Conv2D/ReadVariableOp*^model_18/conv2d_28/BiasAdd/ReadVariableOp)^model_18/conv2d_28/Conv2D/ReadVariableOp*^model_18/conv2d_29/BiasAdd/ReadVariableOp)^model_18/conv2d_29/Conv2D/ReadVariableOp3^model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp<^model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp)^model_18/dense_14/BiasAdd/ReadVariableOp(^model_18/dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2V
)model_18/conv2d_27/BiasAdd/ReadVariableOp)model_18/conv2d_27/BiasAdd/ReadVariableOp2T
(model_18/conv2d_27/Conv2D/ReadVariableOp(model_18/conv2d_27/Conv2D/ReadVariableOp2V
)model_18/conv2d_28/BiasAdd/ReadVariableOp)model_18/conv2d_28/BiasAdd/ReadVariableOp2T
(model_18/conv2d_28/Conv2D/ReadVariableOp(model_18/conv2d_28/Conv2D/ReadVariableOp2V
)model_18/conv2d_29/BiasAdd/ReadVariableOp)model_18/conv2d_29/BiasAdd/ReadVariableOp2T
(model_18/conv2d_29/Conv2D/ReadVariableOp(model_18/conv2d_29/Conv2D/ReadVariableOp2h
2model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp2model_18/conv2d_transpose_6/BiasAdd/ReadVariableOp2z
;model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp;model_18/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2T
(model_18/dense_14/BiasAdd/ReadVariableOp(model_18/dense_14/BiasAdd/ReadVariableOp2R
'model_18/dense_14/MatMul/ReadVariableOp'model_18/dense_14/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?
e
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_95304

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_28_layer_call_fn_95857

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_953402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
(__inference_model_18_layer_call_fn_95726

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_954682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
E
)__inference_reshape_6_layer_call_fn_95799

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_952652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_95322

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? *
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_95243

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_95283

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
C__inference_model_18_layer_call_and_return_conditional_losses_95527

inputs
dense_14_95496
dense_14_95498
conv2d_29_95503
conv2d_29_95505
conv2d_transpose_6_95509
conv2d_transpose_6_95511
conv2d_28_95515
conv2d_28_95517
conv2d_27_95521
conv2d_27_95523
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_95496dense_14_95498*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_952222"
 dense_14/StatefulPartitionedCall?
leaky_re_lu_33/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_952432 
leaky_re_lu_33/PartitionedCall?
reshape_6/PartitionedCallPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_952652
reshape_6/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_29_95503conv2d_29_95505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_952832#
!conv2d_29/StatefulPartitionedCall?
leaky_re_lu_32/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_953042 
leaky_re_lu_32/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_transpose_6_95509conv2d_transpose_6_95511*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_951982,
*conv2d_transpose_6/StatefulPartitionedCall?
leaky_re_lu_31/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_953222 
leaky_re_lu_31/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_31/PartitionedCall:output:0conv2d_28_95515conv2d_28_95517*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_953402#
!conv2d_28/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_953612 
leaky_re_lu_30/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_27_95521conv2d_27_95523*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_953802#
!conv2d_27/StatefulPartitionedCall?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_95809

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?W
?
C__inference_model_18_layer_call_and_return_conditional_losses_95701

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource?
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_6_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/BiasAdd?
leaky_re_lu_33/LeakyRelu	LeakyReludense_14/BiasAdd:output:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_33/LeakyRelux
reshape_6/ShapeShape&leaky_re_lu_33/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_6/Shape?
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stack?
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1?
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2?
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_6/strided_slicex
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/1x
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/2y
reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_6/Reshape/shape/3?
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0"reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shape?
reshape_6/ReshapeReshape&leaky_re_lu_33/LeakyRelu:activations:0 reshape_6/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_6/Reshape?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dreshape_6/Reshape:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/BiasAdd?
leaky_re_lu_32/LeakyRelu	LeakyReluconv2d_29/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_32/LeakyRelu?
conv2d_transpose_6/ShapeShape&leaky_re_lu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_32/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_6/BiasAdd?
leaky_re_lu_31/LeakyRelu	LeakyRelu#conv2d_transpose_6/BiasAdd:output:0*/
_output_shapes
:?????????  *
alpha%???>2
leaky_re_lu_31/LeakyRelu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D&leaky_re_lu_31/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
conv2d_28/BiasAdd?
leaky_re_lu_30/LeakyRelu	LeakyReluconv2d_28/BiasAdd:output:0*0
_output_shapes
:????????? ?*
alpha%???>2
leaky_re_lu_30/LeakyRelu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D&leaky_re_lu_30/LeakyRelu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd~
conv2d_27/TanhTanhconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/Tanh?
IdentityIdentityconv2d_27/Tanh:y:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
C__inference_dense_14_layer_call_and_return_conditional_losses_95761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_95862

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_dense_14_layer_call_fn_95770

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_952222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?W
?
C__inference_model_18_layer_call_and_return_conditional_losses_95639

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource?
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_6_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_14/BiasAdd?
leaky_re_lu_33/LeakyRelu	LeakyReludense_14/BiasAdd:output:0*)
_output_shapes
:???????????*
alpha%???>2
leaky_re_lu_33/LeakyRelux
reshape_6/ShapeShape&leaky_re_lu_33/LeakyRelu:activations:0*
T0*
_output_shapes
:2
reshape_6/Shape?
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_6/strided_slice/stack?
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_1?
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_6/strided_slice/stack_2?
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_6/strided_slicex
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/1x
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_6/Reshape/shape/2y
reshape_6/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_6/Reshape/shape/3?
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0"reshape_6/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_6/Reshape/shape?
reshape_6/ReshapeReshape&leaky_re_lu_33/LeakyRelu:activations:0 reshape_6/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_6/Reshape?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dreshape_6/Reshape:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/BiasAdd?
leaky_re_lu_32/LeakyRelu	LeakyReluconv2d_29/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_32/LeakyRelu?
conv2d_transpose_6/ShapeShape&leaky_re_lu_32/LeakyRelu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0&leaky_re_lu_32/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_6/BiasAdd?
leaky_re_lu_31/LeakyRelu	LeakyRelu#conv2d_transpose_6/BiasAdd:output:0*/
_output_shapes
:?????????  *
alpha%???>2
leaky_re_lu_31/LeakyRelu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D&leaky_re_lu_31/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:????????? ?2
conv2d_28/BiasAdd?
leaky_re_lu_30/LeakyRelu	LeakyReluconv2d_28/BiasAdd:output:0*0
_output_shapes
:????????? ?*
alpha%???>2
leaky_re_lu_30/LeakyRelu?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D&leaky_re_lu_30/LeakyRelu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd~
conv2d_27/TanhTanhconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/Tanh?
IdentityIdentityconv2d_27/Tanh:y:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_30_layer_call_fn_95867

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_953612
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_14_layer_call_and_return_conditional_losses_95222

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d??*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
`
D__inference_reshape_6_layer_call_and_return_conditional_losses_95265

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_27_layer_call_fn_95887

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_953802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_31_layer_call_fn_95838

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_953222
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?"
?
__inference__traced_save_95940
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes~
|: :
d??:??:?@:@: @: : ?:?:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
d??:"

_output_shapes

:??:-)
'
_output_shapes
:?@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :-)
'
_output_shapes
: ?:!

_output_shapes	
:?:-	)
'
_output_shapes
:?: 


_output_shapes
::

_output_shapes
: 
?
?
#__inference_signature_wrapper_95577
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *)
f$R"
 __inference__wrapped_model_951642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?-
?
C__inference_model_18_layer_call_and_return_conditional_losses_95397
input_19
dense_14_95233
dense_14_95235
conv2d_29_95294
conv2d_29_95296
conv2d_transpose_6_95312
conv2d_transpose_6_95314
conv2d_28_95351
conv2d_28_95353
conv2d_27_95391
conv2d_27_95393
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_14_95233dense_14_95235*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_952222"
 dense_14/StatefulPartitionedCall?
leaky_re_lu_33/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_952432 
leaky_re_lu_33/PartitionedCall?
reshape_6/PartitionedCallPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_952652
reshape_6/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_29_95294conv2d_29_95296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_952832#
!conv2d_29/StatefulPartitionedCall?
leaky_re_lu_32/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_953042 
leaky_re_lu_32/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_transpose_6_95312conv2d_transpose_6_95314*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_951982,
*conv2d_transpose_6/StatefulPartitionedCall?
leaky_re_lu_31/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_953222 
leaky_re_lu_31/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_31/PartitionedCall:output:0conv2d_28_95351conv2d_28_95353*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_953402#
!conv2d_28/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_953612 
leaky_re_lu_30/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_27_95391conv2d_27_95393*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_953802#
!conv2d_27/StatefulPartitionedCall?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?
J
.__inference_leaky_re_lu_33_layer_call_fn_95780

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_952432
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?-
?
C__inference_model_18_layer_call_and_return_conditional_losses_95431
input_19
dense_14_95400
dense_14_95402
conv2d_29_95407
conv2d_29_95409
conv2d_transpose_6_95413
conv2d_transpose_6_95415
conv2d_28_95419
conv2d_28_95421
conv2d_27_95425
conv2d_27_95427
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_14_95400dense_14_95402*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_952222"
 dense_14/StatefulPartitionedCall?
leaky_re_lu_33/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_952432 
leaky_re_lu_33/PartitionedCall?
reshape_6/PartitionedCallPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_952652
reshape_6/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_29_95407conv2d_29_95409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_952832#
!conv2d_29/StatefulPartitionedCall?
leaky_re_lu_32/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_953042 
leaky_re_lu_32/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_transpose_6_95413conv2d_transpose_6_95415*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_951982,
*conv2d_transpose_6/StatefulPartitionedCall?
leaky_re_lu_31/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_953222 
leaky_re_lu_31/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_31/PartitionedCall:output:0conv2d_28_95419conv2d_28_95421*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_953402#
!conv2d_28/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_953612 
leaky_re_lu_30/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_27_95425conv2d_27_95427*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_953802#
!conv2d_27/StatefulPartitionedCall?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?
?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_95878

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_6_layer_call_fn_95208

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_951982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?-
?
C__inference_model_18_layer_call_and_return_conditional_losses_95468

inputs
dense_14_95437
dense_14_95439
conv2d_29_95444
conv2d_29_95446
conv2d_transpose_6_95450
conv2d_transpose_6_95452
conv2d_28_95456
conv2d_28_95458
conv2d_27_95462
conv2d_27_95464
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?*conv2d_transpose_6/StatefulPartitionedCall? dense_14/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCallinputsdense_14_95437dense_14_95439*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_952222"
 dense_14/StatefulPartitionedCall?
leaky_re_lu_33/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_952432 
leaky_re_lu_33/PartitionedCall?
reshape_6/PartitionedCallPartitionedCall'leaky_re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_952652
reshape_6/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0conv2d_29_95444conv2d_29_95446*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_952832#
!conv2d_29/StatefulPartitionedCall?
leaky_re_lu_32/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_953042 
leaky_re_lu_32/PartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_32/PartitionedCall:output:0conv2d_transpose_6_95450conv2d_transpose_6_95452*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_951982,
*conv2d_transpose_6/StatefulPartitionedCall?
leaky_re_lu_31/PartitionedCallPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_953222 
leaky_re_lu_31/PartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_31/PartitionedCall:output:0conv2d_28_95456conv2d_28_95458*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_953402#
!conv2d_28/StatefulPartitionedCall?
leaky_re_lu_30/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_953612 
leaky_re_lu_30/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_30/PartitionedCall:output:0conv2d_27_95462conv2d_27_95464*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_953802#
!conv2d_27/StatefulPartitionedCall?
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall+^conv2d_transpose_6/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?-
?
!__inference__traced_restore_95980
file_prefix$
 assignvariableop_dense_14_kernel$
 assignvariableop_1_dense_14_bias'
#assignvariableop_2_conv2d_29_kernel%
!assignvariableop_3_conv2d_29_bias0
,assignvariableop_4_conv2d_transpose_6_kernel.
*assignvariableop_5_conv2d_transpose_6_bias'
#assignvariableop_6_conv2d_28_kernel%
!assignvariableop_7_conv2d_28_bias'
#assignvariableop_8_conv2d_27_kernel%
!assignvariableop_9_conv2d_27_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_29_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_29_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_28_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_28_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_27_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_27_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
(__inference_model_18_layer_call_fn_95491
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_954682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?
`
D__inference_reshape_6_layer_call_and_return_conditional_losses_95794

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_95380

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_95833

inputs
identity~
	LeakyRelu	LeakyReluinputs*A
_output_shapes/
-:+??????????????????????????? *
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:+??????????????????????????? :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_95361

inputs
identity
	LeakyRelu	LeakyReluinputs*B
_output_shapes0
.:,????????????????????????????*
alpha%???>2
	LeakyRelu?
IdentityIdentityLeakyRelu:activations:0*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_32_layer_call_fn_95828

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_953042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_95848

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_95340

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
: ?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
(__inference_model_18_layer_call_fn_95751

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_955272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_95823

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_95198

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_95775

inputs
identityf
	LeakyRelu	LeakyReluinputs*)
_output_shapes
:???????????*
alpha%???>2
	LeakyRelum
IdentityIdentityLeakyRelu:activations:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
(__inference_model_18_layer_call_fn_95550
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_model_18_layer_call_and_return_conditional_losses_955272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????d::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_19
?
~
)__inference_conv2d_29_layer_call_fn_95818

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_952832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_191
serving_default_input_19:0?????????dE
	conv2d_278
StatefulPartitionedCall:0????????? tensorflow/serving/predict:??
?U
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"?R
_tf_keras_network?Q{"class_name": "Functional", "name": "model_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 22528, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_33", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_33", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_6", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 11, 128]}}, "name": "reshape_6", "inbound_nodes": [[["leaky_re_lu_33", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["reshape_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_32", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_32", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_6", "inbound_nodes": [[["leaky_re_lu_32", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_31", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_31", "inbound_nodes": [[["conv2d_transpose_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["leaky_re_lu_31", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_30", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["leaky_re_lu_30", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}, "name": "input_19", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 22528, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["input_19", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_33", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_33", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_6", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 11, 128]}}, "name": "reshape_6", "inbound_nodes": [[["leaky_re_lu_33", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["reshape_6", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_32", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_32", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_6", "inbound_nodes": [[["leaky_re_lu_32", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_31", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_31", "inbound_nodes": [[["conv2d_transpose_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["leaky_re_lu_31", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_30", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["leaky_re_lu_30", 0, 0, {}]]]}], "input_layers": [["input_19", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_19", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 22528, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_33", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_6", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [16, 11, 128]}}}
?	

kernel
 bias
!	variables
"regularization_losses
#trainable_variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 11, 128]}}
?
%	variables
&regularization_losses
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_32", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?


)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 11, 64]}}
?
/	variables
0regularization_losses
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_31", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 22, 32]}}
?
9	variables
:regularization_losses
;trainable_variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_30", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?	

=kernel
>bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [6, 6]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 22, 128]}}
f
0
1
2
 3
)4
*5
36
47
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
 3
)4
*5
36
47
=8
>9"
trackable_list_wrapper
?
Cnon_trainable_variables
Dlayer_metrics

Elayers
	variables
regularization_losses
Flayer_regularization_losses
trainable_variables
Gmetrics
{__call__
z_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
#:!
d??2dense_14/kernel
:??2dense_14/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Hnon_trainable_variables
Ilayer_metrics

Jlayers
	variables
regularization_losses
Klayer_regularization_losses
trainable_variables
Lmetrics
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables
Nlayer_metrics

Olayers
	variables
regularization_losses
Player_regularization_losses
trainable_variables
Qmetrics
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables
Slayer_metrics

Tlayers
	variables
regularization_losses
Ulayer_regularization_losses
trainable_variables
Vmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_29/kernel
:@2conv2d_29/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
Wnon_trainable_variables
Xlayer_metrics

Ylayers
!	variables
"regularization_losses
Zlayer_regularization_losses
#trainable_variables
[metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables
]layer_metrics

^layers
%	variables
&regularization_losses
_layer_regularization_losses
'trainable_variables
`metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1 @2conv2d_transpose_6/kernel
%:# 2conv2d_transpose_6/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
anon_trainable_variables
blayer_metrics

clayers
+	variables
,regularization_losses
dlayer_regularization_losses
-trainable_variables
emetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables
glayer_metrics

hlayers
/	variables
0regularization_losses
ilayer_regularization_losses
1trainable_variables
jmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:) ?2conv2d_28/kernel
:?2conv2d_28/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
knon_trainable_variables
llayer_metrics

mlayers
5	variables
6regularization_losses
nlayer_regularization_losses
7trainable_variables
ometrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables
qlayer_metrics

rlayers
9	variables
:regularization_losses
slayer_regularization_losses
;trainable_variables
tmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?2conv2d_27/kernel
:2conv2d_27/bias
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
unon_trainable_variables
vlayer_metrics

wlayers
?	variables
@regularization_losses
xlayer_regularization_losses
Atrainable_variables
ymetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
 __inference__wrapped_model_95164?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_19?????????d
?2?
(__inference_model_18_layer_call_fn_95491
(__inference_model_18_layer_call_fn_95726
(__inference_model_18_layer_call_fn_95550
(__inference_model_18_layer_call_fn_95751?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_18_layer_call_and_return_conditional_losses_95701
C__inference_model_18_layer_call_and_return_conditional_losses_95431
C__inference_model_18_layer_call_and_return_conditional_losses_95639
C__inference_model_18_layer_call_and_return_conditional_losses_95397?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_14_layer_call_fn_95770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_14_layer_call_and_return_conditional_losses_95761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_33_layer_call_fn_95780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_95775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_6_layer_call_fn_95799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_6_layer_call_and_return_conditional_losses_95794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_29_layer_call_fn_95818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_95809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_32_layer_call_fn_95828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_95823?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv2d_transpose_6_layer_call_fn_95208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_95198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
.__inference_leaky_re_lu_31_layer_call_fn_95838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_95833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_28_layer_call_fn_95857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_95848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_30_layer_call_fn_95867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_95862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_27_layer_call_fn_95887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_95878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_95577input_19"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_95164~
 )*34=>1?.
'?$
"?
input_19?????????d
? "=?:
8
	conv2d_27+?(
	conv2d_27????????? ?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_95878?=>J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv2d_27_layer_call_fn_95887?=>J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
D__inference_conv2d_28_layer_call_and_return_conditional_losses_95848?34I?F
??<
:?7
inputs+??????????????????????????? 
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_conv2d_28_layer_call_fn_95857?34I?F
??<
:?7
inputs+??????????????????????????? 
? "3?0,?????????????????????????????
D__inference_conv2d_29_layer_call_and_return_conditional_losses_95809m 8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_29_layer_call_fn_95818` 8?5
.?+
)?&
inputs??????????
? " ??????????@?
M__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_95198?)*I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_conv2d_transpose_6_layer_call_fn_95208?)*I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
C__inference_dense_14_layer_call_and_return_conditional_losses_95761^/?,
%?"
 ?
inputs?????????d
? "'?$
?
0???????????
? }
(__inference_dense_14_layer_call_fn_95770Q/?,
%?"
 ?
inputs?????????d
? "?????????????
I__inference_leaky_re_lu_30_layer_call_and_return_conditional_losses_95862?J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
.__inference_leaky_re_lu_30_layer_call_fn_95867?J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
I__inference_leaky_re_lu_31_layer_call_and_return_conditional_losses_95833?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
.__inference_leaky_re_lu_31_layer_call_fn_95838I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
I__inference_leaky_re_lu_32_layer_call_and_return_conditional_losses_95823h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_leaky_re_lu_32_layer_call_fn_95828[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
I__inference_leaky_re_lu_33_layer_call_and_return_conditional_losses_95775\1?.
'?$
"?
inputs???????????
? "'?$
?
0???????????
? ?
.__inference_leaky_re_lu_33_layer_call_fn_95780O1?.
'?$
"?
inputs???????????
? "?????????????
C__inference_model_18_layer_call_and_return_conditional_losses_95397?
 )*34=>9?6
/?,
"?
input_19?????????d
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_95431?
 )*34=>9?6
/?,
"?
input_19?????????d
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_95639t
 )*34=>7?4
-?*
 ?
inputs?????????d
p

 
? "-?*
#? 
0????????? 
? ?
C__inference_model_18_layer_call_and_return_conditional_losses_95701t
 )*34=>7?4
-?*
 ?
inputs?????????d
p 

 
? "-?*
#? 
0????????? 
? ?
(__inference_model_18_layer_call_fn_95491{
 )*34=>9?6
/?,
"?
input_19?????????d
p

 
? "2?/+????????????????????????????
(__inference_model_18_layer_call_fn_95550{
 )*34=>9?6
/?,
"?
input_19?????????d
p 

 
? "2?/+????????????????????????????
(__inference_model_18_layer_call_fn_95726y
 )*34=>7?4
-?*
 ?
inputs?????????d
p

 
? "2?/+????????????????????????????
(__inference_model_18_layer_call_fn_95751y
 )*34=>7?4
-?*
 ?
inputs?????????d
p 

 
? "2?/+????????????????????????????
D__inference_reshape_6_layer_call_and_return_conditional_losses_95794c1?.
'?$
"?
inputs???????????
? ".?+
$?!
0??????????
? ?
)__inference_reshape_6_layer_call_fn_95799V1?.
'?$
"?
inputs???????????
? "!????????????
#__inference_signature_wrapper_95577?
 )*34=>=?:
? 
3?0
.
input_19"?
input_19?????????d"=?:
8
	conv2d_27+?(
	conv2d_27????????? 
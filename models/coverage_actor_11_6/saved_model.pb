█ш
к¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8┌в
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	HА*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	HА*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:А*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	А@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@ *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: *
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0

NoOpNoOp
┌d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Хd
valueЛdBИd BБd
╬
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
Ч
(axis
	)gamma
*beta
+moving_mean
,moving_variance
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
Ч
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
 
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
h

Nkernel
Obias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
Ч
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
R
mregularization_losses
n	variables
otrainable_variables
p	keras_api
h

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
R
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
i

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
А	keras_api
n
Бkernel
	Вbias
Гregularization_losses
Д	variables
Еtrainable_variables
Ж	keras_api
n
Зkernel
	Иbias
Йregularization_losses
К	variables
Лtrainable_variables
М	keras_api
n
Нkernel
	Оbias
Пregularization_losses
Р	variables
Сtrainable_variables
Т	keras_api
n
Уkernel
	Фbias
Хregularization_losses
Ц	variables
Чtrainable_variables
Ш	keras_api
n
Щkernel
	Ъbias
Ыregularization_losses
Ь	variables
Эtrainable_variables
Ю	keras_api
 
░
0
1
)2
*3
+4
,5
16
27
<8
=9
>10
?11
D12
E13
N14
O15
U16
V17
W18
X19
]20
^21
g22
h23
q24
r25
{26
|27
Б28
В29
З30
И31
Н32
О33
У34
Ф35
Щ36
Ъ37
А
0
1
)2
*3
14
25
<6
=7
D8
E9
N10
O11
U12
V13
]14
^15
g16
h17
q18
r19
{20
|21
Б22
В23
З24
И25
Н26
О27
У28
Ф29
Щ30
Ъ31
▓
regularization_losses
Яlayer_metrics
аmetrics
	variables
бnon_trainable_variables
 вlayer_regularization_losses
гlayers
trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
▓
 regularization_losses
дlayer_metrics
еmetrics
!	variables
жnon_trainable_variables
 зlayer_regularization_losses
иlayers
"trainable_variables
 
 
 
▓
$regularization_losses
йlayer_metrics
кmetrics
%	variables
лnon_trainable_variables
 мlayer_regularization_losses
нlayers
&trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1
+2
,3

)0
*1
▓
-regularization_losses
оlayer_metrics
пmetrics
.	variables
░non_trainable_variables
 ▒layer_regularization_losses
▓layers
/trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
▓
3regularization_losses
│layer_metrics
┤metrics
4	variables
╡non_trainable_variables
 ╢layer_regularization_losses
╖layers
5trainable_variables
 
 
 
▓
7regularization_losses
╕layer_metrics
╣metrics
8	variables
║non_trainable_variables
 ╗layer_regularization_losses
╝layers
9trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1
>2
?3

<0
=1
▓
@regularization_losses
╜layer_metrics
╛metrics
A	variables
┐non_trainable_variables
 └layer_regularization_losses
┴layers
Btrainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

D0
E1

D0
E1
▓
Fregularization_losses
┬layer_metrics
├metrics
G	variables
─non_trainable_variables
 ┼layer_regularization_losses
╞layers
Htrainable_variables
 
 
 
▓
Jregularization_losses
╟layer_metrics
╚metrics
K	variables
╔non_trainable_variables
 ╩layer_regularization_losses
╦layers
Ltrainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

N0
O1
▓
Pregularization_losses
╠layer_metrics
═metrics
Q	variables
╬non_trainable_variables
 ╧layer_regularization_losses
╨layers
Rtrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1
W2
X3

U0
V1
▓
Yregularization_losses
╤layer_metrics
╥metrics
Z	variables
╙non_trainable_variables
 ╘layer_regularization_losses
╒layers
[trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
▓
_regularization_losses
╓layer_metrics
╫metrics
`	variables
╪non_trainable_variables
 ┘layer_regularization_losses
┌layers
atrainable_variables
 
 
 
▓
cregularization_losses
█layer_metrics
▄metrics
d	variables
▌non_trainable_variables
 ▐layer_regularization_losses
▀layers
etrainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
▓
iregularization_losses
рlayer_metrics
сmetrics
j	variables
тnon_trainable_variables
 уlayer_regularization_losses
фlayers
ktrainable_variables
 
 
 
▓
mregularization_losses
хlayer_metrics
цmetrics
n	variables
чnon_trainable_variables
 шlayer_regularization_losses
щlayers
otrainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
▓
sregularization_losses
ъlayer_metrics
ыmetrics
t	variables
ьnon_trainable_variables
 эlayer_regularization_losses
юlayers
utrainable_variables
 
 
 
▓
wregularization_losses
яlayer_metrics
Ёmetrics
x	variables
ёnon_trainable_variables
 Єlayer_regularization_losses
єlayers
ytrainable_variables
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
▓
}regularization_losses
Їlayer_metrics
їmetrics
~	variables
Ўnon_trainable_variables
 ўlayer_regularization_losses
°layers
trainable_variables
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Б0
В1

Б0
В1
╡
Гregularization_losses
∙layer_metrics
·metrics
Д	variables
√non_trainable_variables
 №layer_regularization_losses
¤layers
Еtrainable_variables
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

З0
И1

З0
И1
╡
Йregularization_losses
■layer_metrics
 metrics
К	variables
Аnon_trainable_variables
 Бlayer_regularization_losses
Вlayers
Лtrainable_variables
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Н0
О1

Н0
О1
╡
Пregularization_losses
Гlayer_metrics
Дmetrics
Р	variables
Еnon_trainable_variables
 Жlayer_regularization_losses
Зlayers
Сtrainable_variables
[Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_8/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

У0
Ф1

У0
Ф1
╡
Хregularization_losses
Иlayer_metrics
Йmetrics
Ц	variables
Кnon_trainable_variables
 Лlayer_regularization_losses
Мlayers
Чtrainable_variables
[Y
VARIABLE_VALUEdense_9/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_9/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Щ0
Ъ1

Щ0
Ъ1
╡
Ыregularization_losses
Нlayer_metrics
Оmetrics
Ь	variables
Пnon_trainable_variables
 Рlayer_regularization_losses
Сlayers
Эtrainable_variables
 
 
*
+0
,1
>2
?3
W4
X5
 
╢
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
11
12
13
14
15
16
17
18
19
20
21
22
23
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

+0
,1
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

>0
?1
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

W0
X1
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
О
serving_default_input_1Placeholder*1
_output_shapes
:         ╚╚*
dtype0*&
shape:         ╚╚
z
serving_default_input_2Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
А	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_32987974
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
б
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpConst*3
Tin,
*2(*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_32989355
д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*2
Tin+
)2'*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_32989481ю╨
щ
н
E__inference_dense_5_layer_call_and_return_conditional_losses_32989124

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╞З
н
C__inference_model_layer_call_and_return_conditional_losses_32988166
inputs_0
inputs_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИв7batch_normalization/AssignMovingAvg/AssignSubVariableOpв9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╗
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         AA 2
conv2d/Relu└
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         !! *
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1╟
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization/Constь
)batch_normalization/AssignMovingAvg/sub/xConst*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)batch_normalization/AssignMovingAvg/sub/xг
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0"batch_normalization/Const:output:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subс
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp┬
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg/sub_1л
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/mul╙
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpЄ
+batch_normalization/AssignMovingAvg_1/sub/xConst*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization/AssignMovingAvg_1/sub/xл
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0"batch_normalization/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subч
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp╬
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization/AssignMovingAvg_1/sub_1╡
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/mulс
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpс
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 *
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         

 2
conv2d_1/Relu╞
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1╒
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_1/ConstЄ
+batch_normalization_1/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_1/AssignMovingAvg/sub/xн
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subч
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp╠
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg/sub_1╡
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mulс
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp°
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x╡
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subэ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp╪
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_1/AssignMovingAvg_1/sub_1┐
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mulя
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpу
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_2/ReluЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOpЗ
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Relu╞
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPoolе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Relu╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1╒
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
batch_normalization_2/ConstЄ
+batch_normalization_2/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_2/AssignMovingAvg/sub/xн
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subч
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp╠
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg/sub_1╡
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mulс
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp°
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x╡
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subэ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp╪
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2/
-batch_normalization_2/AssignMovingAvg_1/sub_1┐
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulя
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
flatten/Constг
flatten/ReshapeReshape*batch_normalization_2/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         @2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/dropout/ConstЭ
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape╠
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2 
dropout/dropout/GreaterEqual/y▐
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/dropout/GreaterEqualЧ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/dropout/CastЪ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/dropout/Mul_1е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_3/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis╚
concatenate/concatConcatV2dropout/dropout/Mul_1:z:0dense_3/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         H2
concatenate/concatж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	HА*
dtype02
dense_4/MatMul/ReadVariableOpб
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_4/MatMulе
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_4/BiasAdd/ReadVariableOpв
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_4/Reluж
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_6/Reluе
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_7/Reluе
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOpЯ
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_8/MatMulд
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOpб
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_8/Reluе
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOpЯ
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddy
dense_9/SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_9/Sigmoid╤
IdentityIdentitydense_9/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:[ W
1
_output_shapes
:         ╚╚
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
С
К
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32986443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ї
F
*__inference_dropout_layer_call_fn_32989060

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
щ$
┌
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32986572

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▓
л
8__inference_batch_normalization_2_layer_call_fn_32988969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870492
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_6_layer_call_fn_32989153

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_329872822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_7_layer_call_fn_32989173

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_329873092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┌
└
&__inference_signature_wrapper_32987974
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_329862942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
╗
a
E__inference_flatten_layer_call_and_return_conditional_losses_32989008

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
щ$
┌
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988756

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
d
E__inference_dropout_layer_call_and_return_conditional_losses_32987156

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
В
i
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32986642

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
№
┬
(__inference_model_layer_call_fn_32987854
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identityИвStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_329877752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
╗
a
E__inference_flatten_layer_call_and_return_conditional_losses_32987136

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Я$
╪
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32986815

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ь
н
E__inference_dense_4_layer_call_and_return_conditional_losses_32987228

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	HА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         H:::O K
'
_output_shapes
:         H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
А
+__inference_conv2d_1_layer_call_fn_32986476

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_329864662
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ
н
E__inference_dense_5_layer_call_and_return_conditional_losses_32987255

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А:::P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╡

о
F__inference_conv2d_1_layer_call_and_return_conditional_losses_32986466

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╚
c
E__inference_dropout_layer_call_and_return_conditional_losses_32987161

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Хq
╗
C__inference_model_layer_call_and_return_conditional_losses_32987380
input_1
input_2
conv2d_32986779
conv2d_32986781 
batch_normalization_32986860 
batch_normalization_32986862 
batch_normalization_32986864 
batch_normalization_32986866
conv2d_1_32986869
conv2d_1_32986871"
batch_normalization_1_32986950"
batch_normalization_1_32986952"
batch_normalization_1_32986954"
batch_normalization_1_32986956
conv2d_2_32986959
conv2d_2_32986961
dense_32986986
dense_32986988
dense_1_32987014
dense_1_32987016"
batch_normalization_2_32987094"
batch_normalization_2_32987096"
batch_normalization_2_32987098"
batch_normalization_2_32987100
dense_2_32987125
dense_2_32987127
dense_3_32987196
dense_3_32987198
dense_4_32987239
dense_4_32987241
dense_5_32987266
dense_5_32987268
dense_6_32987293
dense_6_32987295
dense_7_32987320
dense_7_32987322
dense_8_32987347
dense_8_32987349
dense_9_32987374
dense_9_32987376
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallвdropout/StatefulPartitionedCall·
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_32986779conv2d_32986781*
Tin
2*
Tout
2*/
_output_shapes
:         AA *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_329863062 
conv2d/StatefulPartitionedCallя
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_329863222
max_pooling2d/PartitionedCallШ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_32986860batch_normalization_32986862batch_normalization_32986864batch_normalization_32986866*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868152-
+batch_normalization/StatefulPartitionedCall▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_32986869conv2d_1_32986871*
Tin
2*
Tout
2*/
_output_shapes
:         

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_329864662"
 conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_329864822!
max_pooling2d_1/PartitionedCallи
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_32986950batch_normalization_1_32986952batch_normalization_1_32986954batch_normalization_1_32986956*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869052/
-batch_normalization_1/StatefulPartitionedCall│
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_32986959conv2d_2_32986961*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_329866262"
 conv2d_2/StatefulPartitionedCallэ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_32986986dense_32986988*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_329869752
dense/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_329866422!
max_pooling2d_2/PartitionedCallЦ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_32987014dense_1_32987016*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_329870032!
dense_1/StatefulPartitionedCallи
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_32987094batch_normalization_2_32987096batch_normalization_2_32987098batch_normalization_2_32987100*
Tin	
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870492/
-batch_normalization_2/StatefulPartitionedCallШ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32987125dense_2_32987127*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_329871142!
dense_2/StatefulPartitionedCallф
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_329871362
flatten/PartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871562!
dropout/StatefulPartitionedCallШ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_32987196dense_3_32987198*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_329871852!
dense_3/StatefulPartitionedCallН
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_329872082
concatenate/PartitionedCallХ
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_32987239dense_4_32987241*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_329872282!
dense_4/StatefulPartitionedCallШ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_32987266dense_5_32987268*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_329872552!
dense_5/StatefulPartitionedCallШ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_32987293dense_6_32987295*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_329872822!
dense_6/StatefulPartitionedCallШ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_32987320dense_7_32987322*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_329873092!
dense_7/StatefulPartitionedCallШ
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_32987347dense_8_32987349*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_329873362!
dense_8/StatefulPartitionedCallШ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_32987374dense_9_32987376*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_329873632!
dense_9/StatefulPartitionedCallх
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ь
н
E__inference_dense_4_layer_call_and_return_conditional_losses_32989104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	HА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         H:::O K
'
_output_shapes
:         H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ЮV
г
!__inference__traced_save_32989355
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_dc8f16c2a49d413dacd9450701ad29b0/part2	
Const_1Л
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameц
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*°
valueюBы&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╪
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╠
_input_shapes║
╖: : : : : : : :  : : : : : : @:@:@:@:@:@:@:@:@ : : ::::	HА:А:	А@:@:@ : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	HА:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@ :  

_output_shapes
: :$! 

_output_shapes

: : "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::'

_output_shapes
: 
ч$
╪
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32986412

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▓
л
8__inference_batch_normalization_1_layer_call_fn_32988712

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869052
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
л
C__inference_dense_layer_call_and_return_conditional_losses_32988811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_2_layer_call_and_return_conditional_losses_32987114

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╚
К
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988612

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! :::::W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
М
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32987067

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @:::::W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Б
c
*__inference_dropout_layer_call_fn_32989055

inputs
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╩
М
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988956

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @:::::W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
С
К
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988537

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
А
+__inference_conv2d_2_layer_call_fn_32986636

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_329866262
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_8_layer_call_and_return_conditional_losses_32987336

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
с
~
)__inference_conv2d_layer_call_fn_32986316

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_329863062
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
У
М
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988881

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┤
л
8__inference_batch_normalization_2_layer_call_fn_32988982

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870672
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
·
л
8__inference_batch_normalization_1_layer_call_fn_32988787

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329865722
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┤
л
8__inference_batch_normalization_1_layer_call_fn_32988725

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
М
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32986923

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
В
─
(__inference_model_layer_call_fn_32988476
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *H
_read_only_resource_inputs*
(&	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_329877752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         ╚╚
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ёo
Щ
C__inference_model_layer_call_and_return_conditional_losses_32987775

inputs
inputs_1
conv2d_32987676
conv2d_32987678 
batch_normalization_32987682 
batch_normalization_32987684 
batch_normalization_32987686 
batch_normalization_32987688
conv2d_1_32987691
conv2d_1_32987693"
batch_normalization_1_32987697"
batch_normalization_1_32987699"
batch_normalization_1_32987701"
batch_normalization_1_32987703
conv2d_2_32987706
conv2d_2_32987708
dense_32987711
dense_32987713
dense_1_32987717
dense_1_32987719"
batch_normalization_2_32987722"
batch_normalization_2_32987724"
batch_normalization_2_32987726"
batch_normalization_2_32987728
dense_2_32987731
dense_2_32987733
dense_3_32987738
dense_3_32987740
dense_4_32987744
dense_4_32987746
dense_5_32987749
dense_5_32987751
dense_6_32987754
dense_6_32987756
dense_7_32987759
dense_7_32987761
dense_8_32987764
dense_8_32987766
dense_9_32987769
dense_9_32987771
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCall∙
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32987676conv2d_32987678*
Tin
2*
Tout
2*/
_output_shapes
:         AA *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_329863062 
conv2d/StatefulPartitionedCallя
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_329863222
max_pooling2d/PartitionedCallЪ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_32987682batch_normalization_32987684batch_normalization_32987686batch_normalization_32987688*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868332-
+batch_normalization/StatefulPartitionedCall▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_32987691conv2d_1_32987693*
Tin
2*
Tout
2*/
_output_shapes
:         

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_329864662"
 conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_329864822!
max_pooling2d_1/PartitionedCallк
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_32987697batch_normalization_1_32987699batch_normalization_1_32987701batch_normalization_1_32987703*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869232/
-batch_normalization_1/StatefulPartitionedCall│
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_32987706conv2d_2_32987708*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_329866262"
 conv2d_2/StatefulPartitionedCallю
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_32987711dense_32987713*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_329869752
dense/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_329866422!
max_pooling2d_2/PartitionedCallЦ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_32987717dense_1_32987719*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_329870032!
dense_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_32987722batch_normalization_2_32987724batch_normalization_2_32987726batch_normalization_2_32987728*
Tin	
2*
Tout
2*/
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870672/
-batch_normalization_2/StatefulPartitionedCallШ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32987731dense_2_32987733*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_329871142!
dense_2/StatefulPartitionedCallф
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_329871362
flatten/PartitionedCall╬
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871612
dropout/PartitionedCallШ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_32987738dense_3_32987740*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_329871852!
dense_3/StatefulPartitionedCallЕ
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_329872082
concatenate/PartitionedCallХ
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_32987744dense_4_32987746*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_329872282!
dense_4/StatefulPartitionedCallШ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_32987749dense_5_32987751*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_329872552!
dense_5/StatefulPartitionedCallШ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_32987754dense_6_32987756*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_329872822!
dense_6/StatefulPartitionedCallШ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_32987759dense_7_32987761*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_329873092!
dense_7/StatefulPartitionedCallШ
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_32987764dense_8_32987766*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_329873362!
dense_8/StatefulPartitionedCallШ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_32987769dense_9_32987771*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_329873632!
dense_9/StatefulPartitionedCall├
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
│

м
D__inference_conv2d_layer_call_and_return_conditional_losses_32986306

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           :::i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_3_layer_call_and_return_conditional_losses_32987185

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┼Ц
├
C__inference_model_layer_call_and_return_conditional_losses_32988312
inputs_0
inputs_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource
identityИк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp╗
conv2d/Conv2DConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA *
paddingVALID*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         AA 2
conv2d/Relu└
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:         !! *
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool░
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp╢
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1у
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpщ
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1┌
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpс
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 *
paddingVALID*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         

 2
conv2d_1/Relu╞
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool╢
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp╝
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1щ
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3░
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpу
conv2d_2/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv2d_2/Conv2Dз
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpм
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_2/ReluЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense/MatMul/ReadVariableOpЗ
dense/MatMulMatMulinputs_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Relu╞
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPoolе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Relu╢
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp╝
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1щ
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ш
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3е
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
flatten/Constг
flatten/ReshapeReshape*batch_normalization_2/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*'
_output_shapes
:         @2
flatten/Reshape|
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:         @2
dropout/Identityе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOpЯ
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_3/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis╚
concatenate/concatConcatV2dropout/Identity:output:0dense_3/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         H2
concatenate/concatж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	HА*
dtype02
dense_4/MatMul/ReadVariableOpб
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_4/MatMulе
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_4/BiasAdd/ReadVariableOpв
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
dense_4/Reluж
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_6/Reluе
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_7/Reluе
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_8/MatMul/ReadVariableOpЯ
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_8/MatMulд
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOpб
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_8/BiasAddp
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_8/Reluе
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_9/MatMul/ReadVariableOpЯ
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/MatMulд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpб
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/BiasAddy
dense_9/SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_9/Sigmoidg
IdentityIdentitydense_9/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         :::::::::::::::::::::::::::::::::::::::[ W
1
_output_shapes
:         ╚╚
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
√

*__inference_dense_2_layer_call_fn_32989033

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_329871142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╩
М
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╡

о
F__inference_conv2d_2_layer_call_and_return_conditional_losses_32986626

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                            :::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
░
й
6__inference_batch_normalization_layer_call_fn_32988638

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868332
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
н
E__inference_dense_9_layer_call_and_return_conditional_losses_32987363

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Уq
╗
C__inference_model_layer_call_and_return_conditional_losses_32987590

inputs
inputs_1
conv2d_32987491
conv2d_32987493 
batch_normalization_32987497 
batch_normalization_32987499 
batch_normalization_32987501 
batch_normalization_32987503
conv2d_1_32987506
conv2d_1_32987508"
batch_normalization_1_32987512"
batch_normalization_1_32987514"
batch_normalization_1_32987516"
batch_normalization_1_32987518
conv2d_2_32987521
conv2d_2_32987523
dense_32987526
dense_32987528
dense_1_32987532
dense_1_32987534"
batch_normalization_2_32987537"
batch_normalization_2_32987539"
batch_normalization_2_32987541"
batch_normalization_2_32987543
dense_2_32987546
dense_2_32987548
dense_3_32987553
dense_3_32987555
dense_4_32987559
dense_4_32987561
dense_5_32987564
dense_5_32987566
dense_6_32987569
dense_6_32987571
dense_7_32987574
dense_7_32987576
dense_8_32987579
dense_8_32987581
dense_9_32987584
dense_9_32987586
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCallвdropout/StatefulPartitionedCall∙
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32987491conv2d_32987493*
Tin
2*
Tout
2*/
_output_shapes
:         AA *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_329863062 
conv2d/StatefulPartitionedCallя
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_329863222
max_pooling2d/PartitionedCallШ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_32987497batch_normalization_32987499batch_normalization_32987501batch_normalization_32987503*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868152-
+batch_normalization/StatefulPartitionedCall▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_32987506conv2d_1_32987508*
Tin
2*
Tout
2*/
_output_shapes
:         

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_329864662"
 conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_329864822!
max_pooling2d_1/PartitionedCallи
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_32987512batch_normalization_1_32987514batch_normalization_1_32987516batch_normalization_1_32987518*
Tin	
2*
Tout
2*/
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869052/
-batch_normalization_1/StatefulPartitionedCall│
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_32987521conv2d_2_32987523*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_329866262"
 conv2d_2/StatefulPartitionedCallю
dense/StatefulPartitionedCallStatefulPartitionedCallinputs_1dense_32987526dense_32987528*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_329869752
dense/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_329866422!
max_pooling2d_2/PartitionedCallЦ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_32987532dense_1_32987534*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_329870032!
dense_1/StatefulPartitionedCallи
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_32987537batch_normalization_2_32987539batch_normalization_2_32987541batch_normalization_2_32987543*
Tin	
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870492/
-batch_normalization_2/StatefulPartitionedCallШ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32987546dense_2_32987548*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_329871142!
dense_2/StatefulPartitionedCallф
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_329871362
flatten/PartitionedCallц
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871562!
dropout/StatefulPartitionedCallШ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_32987553dense_3_32987555*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_329871852!
dense_3/StatefulPartitionedCallН
concatenate/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_329872082
concatenate/PartitionedCallХ
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_32987559dense_4_32987561*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_329872282!
dense_4/StatefulPartitionedCallШ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_32987564dense_5_32987566*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_329872552!
dense_5/StatefulPartitionedCallШ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_32987569dense_6_32987571*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_329872822!
dense_6/StatefulPartitionedCallШ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_32987574dense_7_32987576*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_329873092!
dense_7/StatefulPartitionedCallШ
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_32987579dense_8_32987581*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_329873362!
dense_8/StatefulPartitionedCallШ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_32987584dense_9_32987586*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_329873632!
dense_9/StatefulPartitionedCallх
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
Ў
й
6__inference_batch_normalization_layer_call_fn_32988550

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329864122
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
л
C__inference_dense_layer_call_and_return_conditional_losses_32986975

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
л
8__inference_batch_normalization_1_layer_call_fn_32988800

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329866032
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
еж
Е
#__inference__wrapped_model_32986294
input_1
input_2/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource5
1model_batch_normalization_readvariableop_resource7
3model_batch_normalization_readvariableop_1_resourceF
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resourceH
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource7
3model_batch_normalization_1_readvariableop_resource9
5model_batch_normalization_1_readvariableop_1_resourceH
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource7
3model_batch_normalization_2_readvariableop_resource9
5model_batch_normalization_2_readvariableop_1_resourceH
Dmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceJ
Fmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource0
,model_dense_3_matmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource0
,model_dense_5_matmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource0
,model_dense_6_matmul_readvariableop_resource1
-model_dense_6_biasadd_readvariableop_resource0
,model_dense_7_matmul_readvariableop_resource1
-model_dense_7_biasadd_readvariableop_resource0
,model_dense_8_matmul_readvariableop_resource1
-model_dense_8_biasadd_readvariableop_resource0
,model_dense_9_matmul_readvariableop_resource1
-model_dense_9_biasadd_readvariableop_resource
identityИ╝
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp╠
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA *
paddingVALID*
strides
2
model/conv2d/Conv2D│
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp╝
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         AA 2
model/conv2d/BiasAddЗ
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         AA 2
model/conv2d/Relu╥
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*/
_output_shapes
:         !! *
ksize
*
paddingSAME*
strides
2
model/max_pooling2d/MaxPool┬
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02*
(model/batch_normalization/ReadVariableOp╚
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization/ReadVariableOp_1ї
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02;
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp√
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Д
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV3$model/max_pooling2d/MaxPool:output:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2,
*model/batch_normalization/FusedBatchNormV3┬
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp∙
model/conv2d_1/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 *
paddingVALID*
strides
2
model/conv2d_1/Conv2D╣
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp─
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

 2
model/conv2d_1/BiasAddН
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         

 2
model/conv2d_1/Relu╪
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingSAME*
strides
2
model/max_pooling2d_1/MaxPool╚
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02,
*model/batch_normalization_1/ReadVariableOp╬
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02.
,model/batch_normalization_1/ReadVariableOp_1√
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02=
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Т
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&model/max_pooling2d_1/MaxPool:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_1/FusedBatchNormV3┬
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp√
model/conv2d_2/Conv2DConv2D0model/batch_normalization_1/FusedBatchNormV3:y:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
model/conv2d_2/Conv2D╣
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp─
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
model/conv2d_2/BiasAddН
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
model/conv2d_2/Relu▒
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!model/dense/MatMul/ReadVariableOpШ
model/dense/MatMulMatMulinput_2)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
model/dense/Relu╪
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingSAME*
strides
2
model/max_pooling2d_2/MaxPool╖
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02%
#model/dense_1/MatMul/ReadVariableOp╡
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/MatMul╢
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp╣
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/BiasAddВ
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense_1/Relu╚
*model/batch_normalization_2/ReadVariableOpReadVariableOp3model_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02,
*model/batch_normalization_2/ReadVariableOp╬
,model/batch_normalization_2/ReadVariableOp_1ReadVariableOp5model_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02.
,model/batch_normalization_2/ReadVariableOp_1√
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02=
;model/batch_normalization_2/FusedBatchNormV3/ReadVariableOpБ
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02?
=model/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Т
,model/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&model/max_pooling2d_2/MaxPool:output:02model/batch_normalization_2/ReadVariableOp:value:04model/batch_normalization_2/ReadVariableOp_1:value:0Cmodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2.
,model/batch_normalization_2/FusedBatchNormV3╖
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp╖
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/MatMul╢
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp╣
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/BiasAddВ
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense_2/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
model/flatten/Const╗
model/flatten/ReshapeReshape0model/batch_normalization_2/FusedBatchNormV3:y:0model/flatten/Const:output:0*
T0*'
_output_shapes
:         @2
model/flatten/ReshapeО
model/dropout/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*'
_output_shapes
:         @2
model/dropout/Identity╖
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_3/MatMul/ReadVariableOp╖
model/dense_3/MatMulMatMul model/dense_2/Relu:activations:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_3/MatMul╢
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp╣
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_3/BiasAddВ
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense_3/ReluА
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisц
model/concatenate/concatConcatV2model/dropout/Identity:output:0 model/dense_3/Relu:activations:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         H2
model/concatenate/concat╕
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	HА*
dtype02%
#model/dense_4/MatMul/ReadVariableOp╣
model/dense_4/MatMulMatMul!model/concatenate/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/dense_4/MatMul╖
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp║
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
model/dense_4/BiasAddГ
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
model/dense_4/Relu╕
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02%
#model/dense_5/MatMul/ReadVariableOp╖
model/dense_5/MatMulMatMul model/dense_4/Relu:activations:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense_5/MatMul╢
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp╣
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
model/dense_5/BiasAddВ
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
model/dense_5/Relu╖
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02%
#model/dense_6/MatMul/ReadVariableOp╖
model/dense_6/MatMulMatMul model/dense_5/Relu:activations:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_6/MatMul╢
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_6/BiasAdd/ReadVariableOp╣
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_6/BiasAddВ
model/dense_6/ReluRelumodel/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense_6/Relu╖
#model/dense_7/MatMul/ReadVariableOpReadVariableOp,model_dense_7_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_7/MatMul/ReadVariableOp╖
model/dense_7/MatMulMatMul model/dense_6/Relu:activations:0+model/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_7/MatMul╢
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_7/BiasAdd/ReadVariableOp╣
model/dense_7/BiasAddBiasAddmodel/dense_7/MatMul:product:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_7/BiasAddВ
model/dense_7/ReluRelumodel/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense_7/Relu╖
#model/dense_8/MatMul/ReadVariableOpReadVariableOp,model_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_8/MatMul/ReadVariableOp╖
model/dense_8/MatMulMatMul model/dense_7/Relu:activations:0+model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_8/MatMul╢
$model/dense_8/BiasAdd/ReadVariableOpReadVariableOp-model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_8/BiasAdd/ReadVariableOp╣
model/dense_8/BiasAddBiasAddmodel/dense_8/MatMul:product:0,model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_8/BiasAddВ
model/dense_8/ReluRelumodel/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense_8/Relu╖
#model/dense_9/MatMul/ReadVariableOpReadVariableOp,model_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_9/MatMul/ReadVariableOp╖
model/dense_9/MatMulMatMul model/dense_8/Relu:activations:0+model/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_9/MatMul╢
$model/dense_9/BiasAdd/ReadVariableOpReadVariableOp-model_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_9/BiasAdd/ReadVariableOp╣
model/dense_9/BiasAddBiasAddmodel/dense_9/MatMul:product:0,model/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_9/BiasAddЛ
model/dense_9/SigmoidSigmoidmodel/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/dense_9/Sigmoidm
IdentityIdentitymodel/dense_9/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         :::::::::::::::::::::::::::::::::::::::Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
В
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32986482

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ў
┬
(__inference_model_layer_call_fn_32987669
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_329875902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ч$
╪
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988519

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╛
u
I__inference_concatenate_layer_call_and_return_conditional_losses_32989087
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisБ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         H2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         @:         :Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ц
н
E__inference_dense_8_layer_call_and_return_conditional_losses_32989184

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
}
(__inference_dense_layer_call_fn_32988820

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_329869752
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╚
К
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32986833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! :::::W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
н
E__inference_dense_9_layer_call_and_return_conditional_losses_32989204

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_1_layer_call_and_return_conditional_losses_32987003

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╚
c
E__inference_dropout_layer_call_and_return_conditional_losses_32989050

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
№
л
8__inference_batch_normalization_2_layer_call_fn_32988907

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329867632
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_3_layer_call_and_return_conditional_losses_32989071

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_7_layer_call_and_return_conditional_losses_32987309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_6_layer_call_and_return_conditional_losses_32989144

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
О
L
0__inference_max_pooling2d_layer_call_fn_32986328

inputs
identity═
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_329863222
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
єo
Щ
C__inference_model_layer_call_and_return_conditional_losses_32987483
input_1
input_2
conv2d_32987384
conv2d_32987386 
batch_normalization_32987390 
batch_normalization_32987392 
batch_normalization_32987394 
batch_normalization_32987396
conv2d_1_32987399
conv2d_1_32987401"
batch_normalization_1_32987405"
batch_normalization_1_32987407"
batch_normalization_1_32987409"
batch_normalization_1_32987411
conv2d_2_32987414
conv2d_2_32987416
dense_32987419
dense_32987421
dense_1_32987425
dense_1_32987427"
batch_normalization_2_32987430"
batch_normalization_2_32987432"
batch_normalization_2_32987434"
batch_normalization_2_32987436
dense_2_32987439
dense_2_32987441
dense_3_32987446
dense_3_32987448
dense_4_32987452
dense_4_32987454
dense_5_32987457
dense_5_32987459
dense_6_32987462
dense_6_32987464
dense_7_32987467
dense_7_32987469
dense_8_32987472
dense_8_32987474
dense_9_32987477
dense_9_32987479
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallвdense_6/StatefulPartitionedCallвdense_7/StatefulPartitionedCallвdense_8/StatefulPartitionedCallвdense_9/StatefulPartitionedCall·
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_32987384conv2d_32987386*
Tin
2*
Tout
2*/
_output_shapes
:         AA *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_329863062 
conv2d/StatefulPartitionedCallя
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         !! * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_329863222
max_pooling2d/PartitionedCallЪ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_32987390batch_normalization_32987392batch_normalization_32987394batch_normalization_32987396*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868332-
+batch_normalization/StatefulPartitionedCall▒
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_32987399conv2d_1_32987401*
Tin
2*
Tout
2*/
_output_shapes
:         

 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_329864662"
 conv2d_1/StatefulPartitionedCallў
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_329864822!
max_pooling2d_1/PartitionedCallк
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_32987405batch_normalization_1_32987407batch_normalization_1_32987409batch_normalization_1_32987411*
Tin	
2*
Tout
2*/
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_329869232/
-batch_normalization_1/StatefulPartitionedCall│
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_2_32987414conv2d_2_32987416*
Tin
2*
Tout
2*/
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_329866262"
 conv2d_2/StatefulPartitionedCallэ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_32987419dense_32987421*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_329869752
dense/StatefulPartitionedCallў
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_329866422!
max_pooling2d_2/PartitionedCallЦ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_32987425dense_1_32987427*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_329870032!
dense_1/StatefulPartitionedCallк
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_32987430batch_normalization_2_32987432batch_normalization_2_32987434batch_normalization_2_32987436*
Tin	
2*
Tout
2*/
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329870672/
-batch_normalization_2/StatefulPartitionedCallШ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_32987439dense_2_32987441*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_329871142!
dense_2/StatefulPartitionedCallф
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_329871362
flatten/PartitionedCall╬
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_329871612
dropout/PartitionedCallШ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_32987446dense_3_32987448*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_329871852!
dense_3/StatefulPartitionedCallЕ
concatenate/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_329872082
concatenate/PartitionedCallХ
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_32987452dense_4_32987454*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_329872282!
dense_4/StatefulPartitionedCallШ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_32987457dense_5_32987459*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_329872552!
dense_5/StatefulPartitionedCallШ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_32987462dense_6_32987464*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_329872822!
dense_6/StatefulPartitionedCallШ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_32987467dense_7_32987469*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_329873092!
dense_7/StatefulPartitionedCallШ
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_32987472dense_8_32987474*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_329873362!
dense_8/StatefulPartitionedCallШ
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_32987477dense_9_32987479*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_329873632!
dense_9/StatefulPartitionedCall├
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:Z V
1
_output_shapes
:         ╚╚
!
_user_specified_name	input_1:PL
'
_output_shapes
:         
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
¤

*__inference_dense_5_layer_call_fn_32989133

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_5_layer_call_and_return_conditional_losses_329872552
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Б
d
E__inference_dropout_layer_call_and_return_conditional_losses_32989045

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
б$
┌
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32986905

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У
М
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988774

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╡
s
I__inference_concatenate_layer_call_and_return_conditional_losses_32987208

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:         H2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         @:         :O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
А
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_32986322

inputs
identityм
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
В
Z
.__inference_concatenate_layer_call_fn_32989093
inputs_0
inputs_1
identity╡
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:         H* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_329872082
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         H2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         @:         :Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
Е
F
*__inference_flatten_layer_call_fn_32989013

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_329871362
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
ц
н
E__inference_dense_6_layer_call_and_return_conditional_losses_32987282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
─
(__inference_model_layer_call_fn_32988394
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*3
Tin,
*2(*
Tout
2*'
_output_shapes
:         *B
_read_only_resource_inputs$
" 	
 !"#$%&'*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_329875902
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*▌
_input_shapes╦
╚:         ╚╚:         ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         ╚╚
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ц
н
E__inference_dense_7_layer_call_and_return_conditional_losses_32989164

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
°
й
6__inference_batch_normalization_layer_call_fn_32988563

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329864432
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У
М
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32986603

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            :::::i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
б$
┌
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988681

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о
й
6__inference_batch_normalization_layer_call_fn_32988625

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:         !! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_329868152
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У
М
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32986763

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @:::::i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_3_layer_call_fn_32989080

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_329871852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_1_layer_call_and_return_conditional_losses_32988993

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:::O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
·
л
8__inference_batch_normalization_2_layer_call_fn_32988894

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_329867322
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
н
E__inference_dense_2_layer_call_and_return_conditional_losses_32989024

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :::O K
'
_output_shapes
:          
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
рв
ж
$__inference__traced_restore_32989481
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance&
"assignvariableop_6_conv2d_1_kernel$
 assignvariableop_7_conv2d_1_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance'
#assignvariableop_12_conv2d_2_kernel%
!assignvariableop_13_conv2d_2_bias$
 assignvariableop_14_dense_kernel"
assignvariableop_15_dense_bias3
/assignvariableop_16_batch_normalization_2_gamma2
.assignvariableop_17_batch_normalization_2_beta9
5assignvariableop_18_batch_normalization_2_moving_mean=
9assignvariableop_19_batch_normalization_2_moving_variance&
"assignvariableop_20_dense_1_kernel$
 assignvariableop_21_dense_1_bias&
"assignvariableop_22_dense_2_kernel$
 assignvariableop_23_dense_2_bias&
"assignvariableop_24_dense_3_kernel$
 assignvariableop_25_dense_3_bias&
"assignvariableop_26_dense_4_kernel$
 assignvariableop_27_dense_4_bias&
"assignvariableop_28_dense_5_kernel$
 assignvariableop_29_dense_5_bias&
"assignvariableop_30_dense_6_kernel$
 assignvariableop_31_dense_6_bias&
"assignvariableop_32_dense_7_kernel$
 assignvariableop_33_dense_7_bias&
"assignvariableop_34_dense_8_kernel$
 assignvariableop_35_dense_8_bias&
"assignvariableop_36_dense_9_kernel$
 assignvariableop_37_dense_9_bias
identity_39ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1ь
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*°
valueюBы&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names┌
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*о
_output_shapesЫ
Ш::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityО
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2в
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4и
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5м
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ш
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11▓
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ь
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ъ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Щ
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ч
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16и
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_2_gammaIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17з
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_2_betaIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_2_moving_meanIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19▓
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_2_moving_varianceIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ы
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Щ
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ы
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_2_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Щ
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_2_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ы
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_3_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Щ
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_3_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ы
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_4_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Щ
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_4_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ы
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_5_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Щ
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_5_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Ы
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_6_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Щ
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_6_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ы
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_7_kernelIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Щ
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_7_biasIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ы
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_8_kernelIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Щ
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_8_biasIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ы
AssignVariableOp_36AssignVariableOp"assignvariableop_36_dense_9_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Щ
AssignVariableOp_37AssignVariableOp assignvariableop_37_dense_9_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpв
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_38п
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_39"#
identity_39Identity_39:output:0*п
_input_shapesЭ
Ъ: ::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: 
щ$
┌
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32986732

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤

*__inference_dense_4_layer_call_fn_32989113

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_329872282
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:         H::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         H
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
N
2__inference_max_pooling2d_1_layer_call_fn_32986488

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_329864822
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
√

*__inference_dense_8_layer_call_fn_32989193

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_329873362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_1_layer_call_fn_32989002

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_329870032
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
б$
┌
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32987049

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_9_layer_call_fn_32989213

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_329873632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
б$
┌
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988938

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Я$
╪
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988594

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╖
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╛
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ$
┌
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╔
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *дp}?2
Const░
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/x┐
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/subе
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp▐
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1╟
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul╟
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╢
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x╟
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpъ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1╤
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul╒
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp╨
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+                           @2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
N
2__inference_max_pooling2d_2_layer_call_fn_32986648

inputs
identity╧
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_329866422
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ё
serving_default▌
E
input_1:
serving_default_input_1:0         ╚╚
;
input_20
serving_default_input_2:0         ;
dense_90
StatefulPartitionedCall:0         tensorflow/serving/predict:й╛
З╝
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
layer_with_weights-15
layer-23
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+Т&call_and_return_all_conditional_losses
У__call__
Ф_default_save_signature"█┤
_tf_keras_model└┤{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dropout", 0, 0, {}], ["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -3, "maxval": 3, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_9", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 200, 200, 1]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["dropout", 0, 0, {}], ["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -3, "maxval": 3, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["dense_8", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_9", 0, 0]]}}}
¤"·
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
╞


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"Я	
_tf_keras_layerЕ	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "stateful": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 1]}}
┘
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"╚
_tf_keras_layerо{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Х	
(axis
	)gamma
*beta
+moving_mean
,moving_variance
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"┐
_tf_keras_layerе{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
┼	

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
▌
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"╠
_tf_keras_layer▓{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ч	
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"┴
_tf_keras_layerз{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}
├	

Dkernel
Ebias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Ь
_tf_keras_layerВ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}
щ"ц
_tf_keras_input_layer╞{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
▌
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+г&call_and_return_all_conditional_losses
д__call__"╠
_tf_keras_layer▓{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
╔

Nkernel
Obias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"в
_tf_keras_layerИ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
Ч	
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+з&call_and_return_all_conditional_losses
и__call__"┴
_tf_keras_layerз{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 64]}}
╧

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+й&call_and_return_all_conditional_losses
к__call__"и
_tf_keras_layerО{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
┴
cregularization_losses
d	variables
etrainable_variables
f	keras_api
+л&call_and_return_all_conditional_losses
м__call__"░
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
╧

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+н&call_and_return_all_conditional_losses
о__call__"и
_tf_keras_layerО{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
└
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+п&call_and_return_all_conditional_losses
░__call__"п
_tf_keras_layerХ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
╬

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"з
_tf_keras_layerН{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
з
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"Ц
_tf_keras_layer№{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64]}, {"class_name": "TensorShape", "items": [null, 8]}]}
╤

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
А	keras_api
+╡&call_and_return_all_conditional_losses
╢__call__"й
_tf_keras_layerП{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
╫
Бkernel
	Вbias
Гregularization_losses
Д	variables
Еtrainable_variables
Ж	keras_api
+╖&call_and_return_all_conditional_losses
╕__call__"к
_tf_keras_layerР{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
╒
Зkernel
	Иbias
Йregularization_losses
К	variables
Лtrainable_variables
М	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"и
_tf_keras_layerО{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
╒
Нkernel
	Оbias
Пregularization_losses
Р	variables
Сtrainable_variables
Т	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"и
_tf_keras_layerО{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
╘
Уkernel
	Фbias
Хregularization_losses
Ц	variables
Чtrainable_variables
Ш	keras_api
+╜&call_and_return_all_conditional_losses
╛__call__"з
_tf_keras_layerН{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Ё
Щkernel
	Ъbias
Ыregularization_losses
Ь	variables
Эtrainable_variables
Ю	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"├
_tf_keras_layerй{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -3, "maxval": 3, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
 "
trackable_list_wrapper
╨
0
1
)2
*3
+4
,5
16
27
<8
=9
>10
?11
D12
E13
N14
O15
U16
V17
W18
X19
]20
^21
g22
h23
q24
r25
{26
|27
Б28
В29
З30
И31
Н32
О33
У34
Ф35
Щ36
Ъ37"
trackable_list_wrapper
а
0
1
)2
*3
14
25
<6
=7
D8
E9
N10
O11
U12
V13
]14
^15
g16
h17
q18
r19
{20
|21
Б22
В23
З24
И25
Н26
О27
У28
Ф29
Щ30
Ъ31"
trackable_list_wrapper
╙
regularization_losses
Яlayer_metrics
аmetrics
	variables
бnon_trainable_variables
 вlayer_regularization_losses
гlayers
trainable_variables
У__call__
Ф_default_save_signature
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
-
┴serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╡
 regularization_losses
дlayer_metrics
еmetrics
!	variables
жnon_trainable_variables
 зlayer_regularization_losses
иlayers
"trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
$regularization_losses
йlayer_metrics
кmetrics
%	variables
лnon_trainable_variables
 мlayer_regularization_losses
нlayers
&trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
╡
-regularization_losses
оlayer_metrics
пmetrics
.	variables
░non_trainable_variables
 ▒layer_regularization_losses
▓layers
/trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
╡
3regularization_losses
│layer_metrics
┤metrics
4	variables
╡non_trainable_variables
 ╢layer_regularization_losses
╖layers
5trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
7regularization_losses
╕layer_metrics
╣metrics
8	variables
║non_trainable_variables
 ╗layer_regularization_losses
╝layers
9trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
╡
@regularization_losses
╜layer_metrics
╛metrics
A	variables
┐non_trainable_variables
 └layer_regularization_losses
┴layers
Btrainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
╡
Fregularization_losses
┬layer_metrics
├metrics
G	variables
─non_trainable_variables
 ┼layer_regularization_losses
╞layers
Htrainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Jregularization_losses
╟layer_metrics
╚metrics
K	variables
╔non_trainable_variables
 ╩layer_regularization_losses
╦layers
Ltrainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
:@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
╡
Pregularization_losses
╠layer_metrics
═metrics
Q	variables
╬non_trainable_variables
 ╧layer_regularization_losses
╨layers
Rtrainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
U0
V1
W2
X3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
╡
Yregularization_losses
╤layer_metrics
╥metrics
Z	variables
╙non_trainable_variables
 ╘layer_regularization_losses
╒layers
[trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_1/kernel
: 2dense_1/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
╡
_regularization_losses
╓layer_metrics
╫metrics
`	variables
╪non_trainable_variables
 ┘layer_regularization_losses
┌layers
atrainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
cregularization_losses
█layer_metrics
▄metrics
d	variables
▌non_trainable_variables
 ▐layer_regularization_losses
▀layers
etrainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
╡
iregularization_losses
рlayer_metrics
сmetrics
j	variables
тnon_trainable_variables
 уlayer_regularization_losses
фlayers
ktrainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
mregularization_losses
хlayer_metrics
цmetrics
n	variables
чnon_trainable_variables
 шlayer_regularization_losses
щlayers
otrainable_variables
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
╡
sregularization_losses
ъlayer_metrics
ыmetrics
t	variables
ьnon_trainable_variables
 эlayer_regularization_losses
юlayers
utrainable_variables
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
wregularization_losses
яlayer_metrics
Ёmetrics
x	variables
ёnon_trainable_variables
 Єlayer_regularization_losses
єlayers
ytrainable_variables
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
!:	HА2dense_4/kernel
:А2dense_4/bias
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
╡
}regularization_losses
Їlayer_metrics
їmetrics
~	variables
Ўnon_trainable_variables
 ўlayer_regularization_losses
°layers
trainable_variables
╢__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_5/kernel
:@2dense_5/bias
 "
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
0
Б0
В1"
trackable_list_wrapper
╕
Гregularization_losses
∙layer_metrics
·metrics
Д	variables
√non_trainable_variables
 №layer_regularization_losses
¤layers
Еtrainable_variables
╕__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_6/kernel
: 2dense_6/bias
 "
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
0
З0
И1"
trackable_list_wrapper
╕
Йregularization_losses
■layer_metrics
 metrics
К	variables
Аnon_trainable_variables
 Бlayer_regularization_losses
Вlayers
Лtrainable_variables
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
╕
Пregularization_losses
Гlayer_metrics
Дmetrics
Р	variables
Еnon_trainable_variables
 Жlayer_regularization_losses
Зlayers
Сtrainable_variables
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 :2dense_8/kernel
:2dense_8/bias
 "
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
╕
Хregularization_losses
Иlayer_metrics
Йmetrics
Ц	variables
Кnon_trainable_variables
 Лlayer_regularization_losses
Мlayers
Чtrainable_variables
╛__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
 "
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
╕
Ыregularization_losses
Нlayer_metrics
Оmetrics
Ь	variables
Пnon_trainable_variables
 Рlayer_regularization_losses
Сlayers
Эtrainable_variables
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
+0
,1
>2
?3
W4
X5"
trackable_list_wrapper
 "
trackable_list_wrapper
╓
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
11
12
13
14
15
16
17
18
19
20
21
22
23"
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
.
+0
,1"
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
.
>0
?1"
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
.
W0
X1"
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
┌2╫
C__inference_model_layer_call_and_return_conditional_losses_32988312
C__inference_model_layer_call_and_return_conditional_losses_32988166
C__inference_model_layer_call_and_return_conditional_losses_32987380
C__inference_model_layer_call_and_return_conditional_losses_32987483└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
(__inference_model_layer_call_fn_32987669
(__inference_model_layer_call_fn_32988476
(__inference_model_layer_call_fn_32988394
(__inference_model_layer_call_fn_32987854└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
У2Р
#__inference__wrapped_model_32986294ш
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *XвU
SЪP
+К(
input_1         ╚╚
!К
input_2         
г2а
D__inference_conv2d_layer_call_and_return_conditional_losses_32986306╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
И2Е
)__inference_conv2d_layer_call_fn_32986316╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
│2░
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_32986322р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_layer_call_fn_32986328р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ж2Г
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988537
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988594
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988519
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988612┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_layer_call_fn_32988638
6__inference_batch_normalization_layer_call_fn_32988625
6__inference_batch_normalization_layer_call_fn_32988563
6__inference_batch_normalization_layer_call_fn_32988550┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
е2в
F__inference_conv2d_1_layer_call_and_return_conditional_losses_32986466╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
К2З
+__inference_conv2d_1_layer_call_fn_32986476╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
╡2▓
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32986482р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_1_layer_call_fn_32986488р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
О2Л
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988756
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988774
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988681
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988699┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_1_layer_call_fn_32988725
8__inference_batch_normalization_1_layer_call_fn_32988800
8__inference_batch_normalization_1_layer_call_fn_32988712
8__inference_batch_normalization_1_layer_call_fn_32988787┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
е2в
F__inference_conv2d_2_layer_call_and_return_conditional_losses_32986626╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
К2З
+__inference_conv2d_2_layer_call_fn_32986636╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
╡2▓
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32986642р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ъ2Ч
2__inference_max_pooling2d_2_layer_call_fn_32986648р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_32988811в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_layer_call_fn_32988820в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
О2Л
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988881
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988956
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988863
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988938┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_2_layer_call_fn_32988894
8__inference_batch_normalization_2_layer_call_fn_32988907
8__inference_batch_normalization_2_layer_call_fn_32988982
8__inference_batch_normalization_2_layer_call_fn_32988969┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_32988993в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_1_layer_call_fn_32989002в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_32989008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_layer_call_fn_32989013в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_32989024в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_2_layer_call_fn_32989033в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚2┼
E__inference_dropout_layer_call_and_return_conditional_losses_32989050
E__inference_dropout_layer_call_and_return_conditional_losses_32989045┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_layer_call_fn_32989060
*__inference_dropout_layer_call_fn_32989055┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_dense_3_layer_call_and_return_conditional_losses_32989071в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_3_layer_call_fn_32989080в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_concatenate_layer_call_and_return_conditional_losses_32989087в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_concatenate_layer_call_fn_32989093в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_4_layer_call_and_return_conditional_losses_32989104в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_4_layer_call_fn_32989113в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_5_layer_call_and_return_conditional_losses_32989124в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_5_layer_call_fn_32989133в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_6_layer_call_and_return_conditional_losses_32989144в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_6_layer_call_fn_32989153в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_7_layer_call_and_return_conditional_losses_32989164в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_7_layer_call_fn_32989173в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_8_layer_call_and_return_conditional_losses_32989184в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_8_layer_call_fn_32989193в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_dense_9_layer_call_and_return_conditional_losses_32989204в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_9_layer_call_fn_32989213в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
<B:
&__inference_signature_wrapper_32987974input_1input_2ё
#__inference__wrapped_model_32986294╔0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪbв_
XвU
SЪP
+К(
input_1         ╚╚
!К
input_2         
к "1к.
,
dense_9!К
dense_9         ╔
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988681r<=>?;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ ╔
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988699r<=>?;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ ю
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988756Ц<=>?MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ю
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_32988774Ц<=>?MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ б
8__inference_batch_normalization_1_layer_call_fn_32988712e<=>?;в8
1в.
(К%
inputs          
p
к " К          б
8__inference_batch_normalization_1_layer_call_fn_32988725e<=>?;в8
1в.
(К%
inputs          
p 
к " К          ╞
8__inference_batch_normalization_1_layer_call_fn_32988787Й<=>?MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╞
8__inference_batch_normalization_1_layer_call_fn_32988800Й<=>?MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ю
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988863ЦUVWXMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ю
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988881ЦUVWXMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ╔
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988938rUVWX;в8
1в.
(К%
inputs         @
p
к "-в*
#К 
0         @
Ъ ╔
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_32988956rUVWX;в8
1в.
(К%
inputs         @
p 
к "-в*
#К 
0         @
Ъ ╞
8__inference_batch_normalization_2_layer_call_fn_32988894ЙUVWXMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @╞
8__inference_batch_normalization_2_layer_call_fn_32988907ЙUVWXMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @б
8__inference_batch_normalization_2_layer_call_fn_32988969eUVWX;в8
1в.
(К%
inputs         @
p
к " К         @б
8__inference_batch_normalization_2_layer_call_fn_32988982eUVWX;в8
1в.
(К%
inputs         @
p 
к " К         @ь
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988519Ц)*+,MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ь
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988537Ц)*+,MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ╟
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988594r)*+,;в8
1в.
(К%
inputs         !! 
p
к "-в*
#К 
0         !! 
Ъ ╟
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_32988612r)*+,;в8
1в.
(К%
inputs         !! 
p 
к "-в*
#К 
0         !! 
Ъ ─
6__inference_batch_normalization_layer_call_fn_32988550Й)*+,MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ─
6__inference_batch_normalization_layer_call_fn_32988563Й)*+,MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            Я
6__inference_batch_normalization_layer_call_fn_32988625e)*+,;в8
1в.
(К%
inputs         !! 
p
к " К         !! Я
6__inference_batch_normalization_layer_call_fn_32988638e)*+,;в8
1в.
(К%
inputs         !! 
p 
к " К         !! ╤
I__inference_concatenate_layer_call_and_return_conditional_losses_32989087ГZвW
PвM
KЪH
"К
inputs/0         @
"К
inputs/1         
к "%в"
К
0         H
Ъ и
.__inference_concatenate_layer_call_fn_32989093vZвW
PвM
KЪH
"К
inputs/0         @
"К
inputs/1         
к "К         H█
F__inference_conv2d_1_layer_call_and_return_conditional_losses_32986466Р12IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                            
Ъ │
+__inference_conv2d_1_layer_call_fn_32986476Г12IвF
?в<
:К7
inputs+                            
к "2К/+                            █
F__inference_conv2d_2_layer_call_and_return_conditional_losses_32986626РDEIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ │
+__inference_conv2d_2_layer_call_fn_32986636ГDEIвF
?в<
:К7
inputs+                            
к "2К/+                           @┘
D__inference_conv2d_layer_call_and_return_conditional_losses_32986306РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ ▒
)__inference_conv2d_layer_call_fn_32986316ГIвF
?в<
:К7
inputs+                           
к "2К/+                            е
E__inference_dense_1_layer_call_and_return_conditional_losses_32988993\]^/в,
%в"
 К
inputs         @
к "%в"
К
0          
Ъ }
*__inference_dense_1_layer_call_fn_32989002O]^/в,
%в"
 К
inputs         @
к "К          е
E__inference_dense_2_layer_call_and_return_conditional_losses_32989024\gh/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ }
*__inference_dense_2_layer_call_fn_32989033Ogh/в,
%в"
 К
inputs          
к "К         е
E__inference_dense_3_layer_call_and_return_conditional_losses_32989071\qr/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ }
*__inference_dense_3_layer_call_fn_32989080Oqr/в,
%в"
 К
inputs         
к "К         ж
E__inference_dense_4_layer_call_and_return_conditional_losses_32989104]{|/в,
%в"
 К
inputs         H
к "&в#
К
0         А
Ъ ~
*__inference_dense_4_layer_call_fn_32989113P{|/в,
%в"
 К
inputs         H
к "К         Аи
E__inference_dense_5_layer_call_and_return_conditional_losses_32989124_БВ0в-
&в#
!К
inputs         А
к "%в"
К
0         @
Ъ А
*__inference_dense_5_layer_call_fn_32989133RБВ0в-
&в#
!К
inputs         А
к "К         @з
E__inference_dense_6_layer_call_and_return_conditional_losses_32989144^ЗИ/в,
%в"
 К
inputs         @
к "%в"
К
0          
Ъ 
*__inference_dense_6_layer_call_fn_32989153QЗИ/в,
%в"
 К
inputs         @
к "К          з
E__inference_dense_7_layer_call_and_return_conditional_losses_32989164^НО/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ 
*__inference_dense_7_layer_call_fn_32989173QНО/в,
%в"
 К
inputs          
к "К         з
E__inference_dense_8_layer_call_and_return_conditional_losses_32989184^УФ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ 
*__inference_dense_8_layer_call_fn_32989193QУФ/в,
%в"
 К
inputs         
к "К         з
E__inference_dense_9_layer_call_and_return_conditional_losses_32989204^ЩЪ/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ 
*__inference_dense_9_layer_call_fn_32989213QЩЪ/в,
%в"
 К
inputs         
к "К         г
C__inference_dense_layer_call_and_return_conditional_losses_32988811\NO/в,
%в"
 К
inputs         
к "%в"
К
0         @
Ъ {
(__inference_dense_layer_call_fn_32988820ONO/в,
%в"
 К
inputs         
к "К         @е
E__inference_dropout_layer_call_and_return_conditional_losses_32989045\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ е
E__inference_dropout_layer_call_and_return_conditional_losses_32989050\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ }
*__inference_dropout_layer_call_fn_32989055O3в0
)в&
 К
inputs         @
p
к "К         @}
*__inference_dropout_layer_call_fn_32989060O3в0
)в&
 К
inputs         @
p 
к "К         @й
E__inference_flatten_layer_call_and_return_conditional_losses_32989008`7в4
-в*
(К%
inputs         @
к "%в"
К
0         @
Ъ Б
*__inference_flatten_layer_call_fn_32989013S7в4
-в*
(К%
inputs         @
к "К         @Ё
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32986482ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_1_layer_call_fn_32986488СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Ё
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32986642ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╚
2__inference_max_pooling2d_2_layer_call_fn_32986648СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_32986322ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_layer_call_fn_32986328СRвO
HвE
CК@
inputs4                                    
к ";К84                                    Н
C__inference_model_layer_call_and_return_conditional_losses_32987380┼0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪjвg
`в]
SЪP
+К(
input_1         ╚╚
!К
input_2         
p

 
к "%в"
К
0         
Ъ Н
C__inference_model_layer_call_and_return_conditional_losses_32987483┼0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪjвg
`в]
SЪP
+К(
input_1         ╚╚
!К
input_2         
p 

 
к "%в"
К
0         
Ъ П
C__inference_model_layer_call_and_return_conditional_losses_32988166╟0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪlвi
bв_
UЪR
,К)
inputs/0         ╚╚
"К
inputs/1         
p

 
к "%в"
К
0         
Ъ П
C__inference_model_layer_call_and_return_conditional_losses_32988312╟0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪlвi
bв_
UЪR
,К)
inputs/0         ╚╚
"К
inputs/1         
p 

 
к "%в"
К
0         
Ъ х
(__inference_model_layer_call_fn_32987669╕0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪjвg
`в]
SЪP
+К(
input_1         ╚╚
!К
input_2         
p

 
к "К         х
(__inference_model_layer_call_fn_32987854╕0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪjвg
`в]
SЪP
+К(
input_1         ╚╚
!К
input_2         
p 

 
к "К         ч
(__inference_model_layer_call_fn_32988394║0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪlвi
bв_
UЪR
,К)
inputs/0         ╚╚
"К
inputs/1         
p

 
к "К         ч
(__inference_model_layer_call_fn_32988476║0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪlвi
bв_
UЪR
,К)
inputs/0         ╚╚
"К
inputs/1         
p 

 
к "К         Е
&__inference_signature_wrapper_32987974┌0)*+,12<=>?DENO]^UVWXghqr{|БВЗИНОУФЩЪsвp
в 
iкf
6
input_1+К(
input_1         ╚╚
,
input_2!К
input_2         "1к.
,
dense_9!К
dense_9         
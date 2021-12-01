batch_size = number of samples taken each epoch to train
test_size = how mcu data is for test data
featurewise_center = aya miaane ye data ha ro 0 kone ta biofte center ( chinesh e tasaavir) ya na
geaturewise_std_normalization = standardize datas by devision by STD
regularization = L2(y method e regularization) = regular mikone ta dadeye part nadashte bashim (ye joor regression)
earlyStopping = stop kardan e learning vaghti loss starts to increase
ReduceLROnPlateau = decreases learning rate when when stoped to improving 
verbose = 0(silent) or 1(show messages) .
ModelCheckpoint = save model at every checkpoint
fit_generator = train model
tensor = general matrix 
expand_dims ==> create an extra dimension in a specified index (-1 means the last index)
dummies = a 2D matrix , rows indicates 0..n-1   and cols indicates not repeated characters.   default value of cell is 0 ; 1 is for exsistence
suppose a(2)(4) as a(2,4)  although you know the difference
Conv2D = create a kernel .   first argument is dimension of output space, second is kernel size , thirs is stride
Activation = activation function
maxPooling2D = put maximum of every subregion in the output
padding same means the equality of input & output size
GloabalAveragePooling = take average of each dimension as the value o output in that dimension 
metrics = which metric to be evaluated during training
'.summary'  prints the information of the model
patience is number of epochs with no improvments
factor is the coEfficient of learning rate after patience

# -*- coding: utf-8 -*-

import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

from caffe import layers as L
from caffe import params as P

from caffe.proto import caffe_pb2



weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2



def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
    
def setEnvCifar100():

    envlist = {}
    envlist['caffe_root'] = '/Users/donchan/caffe/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    envlist['caffe_model'] = 'cnn_snapshot_iter_150000.caffemodel'
    envlist['deploy'] = 'cnn_deploy.prototxt'
    envlist['meannpy'] = 'mean.npy'
    envlist['solver'] = 'cnn_solver_h5.prototxt'
    envlist['meanfile'] =  '/Users/donchan/Documents/Statistical_Mechanics/caffe/cifar/mean.binaryproto'
    #envlist['imagefile'] = lmdb_img # after loading image with caffe io load_image

    envlist['weights'] = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    envlist['imagenet_label_file'] = 'data/ilsvrc12/synset_words.txt'
    envlist['style_name'] = 'examples/finetune_flickr_style/style_names.txt'
    envlist['train'] = 'models/bvlc_reference_caffenet/train.prototxt'
    
    return envlist


def process(envlist):
    w_file = os.path.join(envlist['caffe_root'],envlist['weights'])    
    print "[ weights file ]",w_file
    assert os.path.exists(w_file)
    return w_file

def process2(envlist):

    image_label_file = os.path.join(envlist['caffe_root'],envlist['imagenet_label_file'])    
    imagenet_labels = list(np.loadtxt(image_label_file , str, delimiter='\t'))
    assert len(imagenet_labels) == 1000
    print 'Loaded ImageNet labels:\n', '\n'.join(imagenet_labels[:10] + ['...'])

    # Load style labels to style_labels
    style_label_file = os.path.join(envlist['caffe_root'],envlist['style_name'])    
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    
    #NUM_STYLE_LABELS = 5
    #if NUM_STYLE_LABELS > 0:
    #    style_labels = style_labels[:NUM_STYLE_LABELS]
    #print '\nLoaded style labels:\n', ', '.join(style_labels)

    return imagenet_labels,style_labels

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    
    envlist = setEnvCifar100()
    filename = os.path.join(envlist['caffe_root'],envlist['train'])    
    
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    #with tempfile.NamedTemporaryFile(delete=False) as f:
    #    f.write(str(n.to_proto()))
    #    return f.name
    with open(filename, 'w') as f:
        f.write(str( n.to_proto()  ) )
        return f.name

def style_net(train=True, learn_all=False, subset=None):

    envlist = setEnvCifar100()
    caffe_root = envlist['caffe_root']
    NUM_STYLE_LABELS = 5
    
    if subset is None:
        subset = 'train_osx' if train else 'test_osx'

    subset_ =  'data/flickr_style/%s.txt' % subset
    source = os.path.join(caffe_root,subset_)

    mean_file=os.path.join(caffe_root,'data/ilsvrc12/imagenet_mean.binaryproto')
    transform_param = dict(mirror=train, crop_size=227,mean_file=mean_file )

    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)

    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',
                    learn_all=learn_all)

def setcaffenet(weights):
    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    imagenet_net_filename = caffenet(data=dummy_data, train=False)
    print "[ generated trained file name ]", imagenet_net_filename
    imagenet_net = caffe.Net(imagenet_net_filename, weights, caffe.TEST)

    return imagenet_net

def untrained(weights,imagenet_labels,style_labels):
    caffenet_filename = style_net(train=False, subset='train_osx')
    print "[ untrained caffe net ]", caffenet_filename    
    untrained_style_net = caffe.Net(caffenet_filename,
                                weights, caffe.TEST)
    untrained_style_net.forward()
    
    style_data_batch = untrained_style_net.blobs['data'].data.copy()
    style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)    



    batch_index = 8
    image = style_data_batch[batch_index]

    plt.imshow(deprocess_net_image(image))
    print 'actual label =', style_labels[style_label_batch[batch_index]]    
    print "[ style batch label ]", style_label_batch[batch_index]
    
    return image,untrained_style_net

def disp_preds(net, image, labels, k=5, name='ImageNet'):
    # input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image, imagenet_labels):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image, style_labels):
    disp_preds(net, image, style_labels, name='style')    


def solver(train_net_path, scratch=False, test_net_path=None, base_lr=0.001):

    envlist = setEnvCifar100()

    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    
    if scratch:
        proto_ = 'models/finetune_flickr_style/mysolver_scratch.prototxt'
        style_ = 'models/finetune_flickr_style/myfinetune_flickr_style_scrach'
    else:
        proto_ = 'models/finetune_flickr_style/mysolver.prototxt'
        style_ = 'models/finetune_flickr_style/myfinetune_flickr_style'
        
    s.snapshot_prefix = os.path.join(envlist['caffe_root'],style_)
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    filename = os.path.join(envlist['caffe_root'],proto_)
                            
    # Write the solver to a temporary file and return its filename.
    with open(filename, 'w') as f:
#    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name

def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            
            print "loss",name,s.net.blobs["loss"].data.copy()             
            print "acc",name,s.net.blobs["acc"].data.copy()             
            
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = '/Users/donchan/caffe/caffe/models/finetune_flickr_style'
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    
    return loss, acc, weights

def mySolved(weights):
# Reset style_solver as before.    
    style_solver_filename = solver(style_net(train=True))
    print "[ style solver name]", style_solver_filename
    style_solver = caffe.get_solver(style_solver_filename)
    style_solver.net.copy_from(weights)    

    # For reference, we also create a solver that isn't initialized from
    # the pretrained ImageNet weights.
    scratch_style_solver_filename = solver(style_net(train=True),True)
    scratch_style_solver = caffe.get_solver(scratch_style_solver_filename)

    niter = 100
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', style_solver),
           ('scratch', scratch_style_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print 'Done.'

    train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
    train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
    style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(211)
    ax1.plot(np.vstack([train_loss, scratch_train_loss]).T)
    ax1.set_xlabel('Iteration #')
    ax1.set_ylabel('Loss')

    plt.show()

def makeSolver(weights):
# Reset style_solver as before.    
    style_solver_filename = solver(style_net(train=True))
    print "[ style solver name]", style_solver_filename


def main():

    envlist = setEnvCifar100()
        
    weights = process(envlist)    
    imagenet_labels,style_labels = process2(envlist)
    imagenet_net = setcaffenet(weights)
    
    image,untrained_style_net = untrained(weights,imagenet_labels,style_labels)
    
#    disp_imagenet_preds(imagenet_net, image, imagenet_labels)
#    disp_style_preds(untrained_style_net, image, style_labels)

#    diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
#    error = (diff ** 2).sum()
#    assert error < 1e-8

    #makeSolver(weights)
    
if __name__ == "__main__":
    main()    
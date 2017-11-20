'''
Created on Sep 3, 2017

@author: georgeretsi

do not support multi-gpu yet. needs thread manipulation
- works on GW + IAM
- new way to load dataset
- augmentation with dataloader
- Hardcoded selections (augmentation - default:YES, load pretrained model with hardcoded name...
- do not normalize with respect to iter size (or batch size) for speed
- add fixed size selection (not hardcoded)
- save and load hardcoded name 'PHOCNet.pt'
'''
import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import tqdm

import copy
from cnn_ws_experiments.datasets.iam_alt import IAMDataset
from cnn_ws_experiments.datasets.gw_alt import GWDataset

#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation
from cnn_ws.losses.cosine_loss import CosineLoss

from cnn_ws.models.myphocnet import PHOCNet
from cnn_ws.evaluation.retrieval import map_from_feature_matrix, map_from_query_test_feature_matrices
from torch.utils.data.dataloader import DataLoaderIter
from torch.utils.data.sampler import WeightedRandomSampler

from cnn_ws.utils.save_load import my_torch_save, my_torch_load

def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')
    # argument parsing
    parser = argparse.ArgumentParser()    
    # - train arguments
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser, default='60000:1e-4,100000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=10,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    #parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default=0,
    #                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # - experiment arguments
    parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN. Default: 26')
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc', 'spoc', 'dctow', 'phoc-ppmi', 'phoc-pruned'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc, spoc, phoc-ppmi, phoc-pruned. Default: phoc')
    parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default=None ,
                        help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
    parser.add_argument('--dataset', '-ds', required=True, choices=['gw','iam'], default= 'gw',
                        help='The dataset to be trained on')
    args = parser.parse_args()

    # sanity checks
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).iteritems():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader
    #TODO: add augmentation
    logger.info('Loading dataset %s...', args.dataset)
    if args.dataset == 'gw':
        train_set = GWDataset(gw_root_dir='../../../phocnet-pytorch-master/data/gw',
                              cv_split_method='almazan',
                              cv_split_idx=1,
                              image_extension='.tif',
                              embedding=args.embedding_type,
                              phoc_unigram_levels=args.phoc_unigram_levels,
                              fixed_image_size=args.fixed_image_size,
                              min_image_width_height=args.min_image_width_height)


    if args.dataset == 'iam':
        train_set = IAMDataset(gw_root_dir='../../../phocnet-pytorch-master/data/IAM',
                               image_extension='.png',
                               embedding=args.embedding_type,
                               phoc_unigram_levels=args.phoc_unigram_levels,
                               fixed_image_size=args.fixed_image_size,
                               min_image_width_height=args.min_image_width_height)

    test_set = copy.copy(train_set)

    train_set.mainLoader(partition='train')
    test_set.mainLoader(partition='test', transforms=None)

    # augmentation using data sampler
    n_train_images = 500000
    augmentation = True

    if augmentation:
        train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images),
                                  batch_size=args.batch_size,
                                  num_workers=8)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=8)

    train_loader_iter = DataLoaderIter(loader=train_loader)
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8)
    # load CNN
    logger.info('Preparing PHOCNet...')

    cnn = PHOCNet(n_out=train_set[0][1].shape[0],
                  input_channels=1,
                  gpp_type='gpp',
                  pooling_levels=([1], [5]))

    cnn.init_weights()

    ## pre-trained!!!!
    load_pretrained = True
    if load_pretrained:
        #cnn.load_state_dict(torch.load('PHOCNet.pt', map_location=lambda storage, loc: storage))
        my_torch_load(cnn, 'PHOCNet.pt')

    loss_selection = 'BCE' # or 'cosine'
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')

    # move CNN to GPU
    if args.gpu_id is not None:
        if len(args.gpu_id) > 1:
            cnn = nn.DataParallel(cnn, device_ids=args.gpu_id)
            cnn.cuda()
        else:
            cnn.cuda(args.gpu_id[0])

    # run training
    lr_cnt = 0
    max_iters = args.learning_rate_step[-1][0]
    if args.solver_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), args.learning_rate_step[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate_step[0][1],
                                    weight_decay=args.weight_decay)


    optimizer.zero_grad()
    logger.info('Training:')
    for iter_idx in range(max_iters):
        if iter_idx % args.test_interval == 0: # and iter_idx > 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            evaluate_cnn(cnn=cnn,
                         dataset_loader=test_loader,
                         args=args)        
        for _ in range(args.iter_size):
            if train_loader_iter.batches_outstanding == 0:
                train_loader_iter = DataLoaderIter(loader=train_loader)
                logger.info('Resetting data loader')
            word_img, embedding, _, _ = train_loader_iter.next()
            if args.gpu_id is not None:
                if len(args.gpu_id) > 1:
                    word_img = word_img.cuda()
                    embedding = embedding.cuda()
                else:
                    word_img = word_img.cuda(args.gpu_id[0])
                    embedding = embedding.cuda(args.gpu_id[0])

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            output = cnn(word_img)
            ''' BCEloss ??? '''
            loss_val = loss(output, embedding)*args.batch_size
            loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # mean runing errors??
        if (iter_idx+1) % args.display == 0:
            logger.info('Iteration %*d: %f', len(str(max_iters)), iter_idx+1, loss_val.data[0])

        # change lr
        if (iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx+1) != max_iters:
            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate_step[lr_cnt][1]

        #if (iter_idx + 1) % 10000 == 0:
        #    torch.save(cnn.state_dict(), 'PHOCNet.pt')
            # .. to load your previously training model:
            #cnn.load_state_dict(torch.load('PHOCNet.pt'))

    #torch.save(cnn.state_dict(), 'PHOCNet.pt')
    my_torch_save(cnn, 'PHOCNet.pt')


def evaluate_cnn(cnn, dataset_loader, args):
    logger = logging.getLogger('PHOCNet-Experiment::test')
    # set the CNN in eval mode
    cnn.eval()
    logger.info('Computing net output:')
    qry_ids = [] #np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm.tqdm(dataset_loader)):
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            #word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''
        output = torch.sigmoid(cnn(word_img))
        #output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0,0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    '''
    # find queries

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed
    
    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    ave_precs_qbe = map_from_query_test_feature_matrices(query_features = qry_outputs,
                                                         test_features=outputs,
                                                         query_labels = qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0])*100)



    # clean up -> set CNN in train mode again
    cnn.train()

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()

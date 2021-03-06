from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', 0)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [
                    data['fc_feats'], 
                    data['att_feats'], 
                    data['scene_feats'], 
                    data['labels'], 
                    data['masks'], 
                    data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, scene_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(
                        model(
                            fc_feats, 
                            att_feats, 
                            scene_feats,
                            labels, 
                            att_masks), 
                        labels[:,1:], 
                        masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['scene_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]
                if data.get('att_masks', None) is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, scene_feats, att_masks  = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            res = model(
                    fc_feats, 
                    att_feats, 
                    scene_feats,
                    att_masks, 
                    opt=eval_kwargs, 
                    mode='sample_viz'
                    )
            seq = res[0].data
            print(len(res))
            print(len(res[-1])) # weight list = [weight_1 weight_2, ...]
            print(len(res[-1][-1])) # weight = [conv_weight, vc_weight]
            print(len(res[-1][-1][-1])) # conv_weight
            print(type(res[-1][-1][-1]), res[-1][-1][0].shape) # conv_weight
            print(type(res[-1][-1][-1]), res[-1][-1][1].shape) # conv_weight
            att_list = res[-1]
            print(type(seq), seq.shape)
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join(
                    [
                        utils.decode_sequence(
                            loader.get_vocab(), 
                            _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + \
                        os.path.join(
                            eval_kwargs['image_root'], 
                            data['infos'][k]['file_path']) + \
                        '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s (%s): %s' %(data['infos'][k]['file_path'], entry['image_id'], entry['caption']))

            for t in range(len(att_list)):
                conv_w, vc_w = att_list[t]
                output_path = os.path.join(
                        'attMap',
                        str(entry['image_id']) + '_' + data['infos'][k]['file_path'].split('/')[-1])
                # print(output_path)
                np.save(output_path+'_conv_%s'%t, conv_w[k].data.cpu().numpy())
                np.save(output_path+'_vc_%s'%t, vc_w[k].data.cpu().numpy())

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            print("break")
            break

    lang_stats = None
    if lang_eval == 1:
        print("language eval begin")
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
        print("language eval done")

    # Switch back to training mode
    model.train()
    print("# Switch back to training mode")
    return loss_sum/loss_evals, predictions, lang_stats

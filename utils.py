import argparse
import os
import torch
import torch.nn.functional as F
import yaml

import numpy as np


def read_yml(dataset:str):
    with open('corpus/classes_'+dataset+'.yml', 'r', encoding='utf-8') as f:
        file_content = f.read()
        content = yaml.safe_load(file_content)
    return content


def cosine_similarity_1d(x,y):
    x = F.normalize(x,p=2,dim=-1)
    y = F.normalize(y,p=2,dim=-1)
    return F.cosine_similarity(x,y, dim=0)

def cosine_similarity(x,y):
    assert x.shape[-1] == y.shape[-1]
    # x= x/x.norm()
    # y = y/y.norm()
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x.to(torch.float16)@y.to(torch.float16).transpose(-2,-1)

def euclidean_dist( x, y, normalize=False):
    # x: N x D
    # y: M x D
    if normalize:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    # return x@y.T

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    x = x.to(y.dtype)
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)#correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def read_yaml():
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
    

hmdb_c = ['brush hair', 'catch', 'chew', 'clap', 'climb', 'climb stairs', 'dive', 'draw sword', 'dribble', 'drink', 'fall floor', 'flic flac', 'handstand', 'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'ride bike', 'ride horse', 'shake hands', 'shoot bow', 'situp', 'stand', 'sword', 'sword exercise', 'throw', 'turn', 'walk', 'wave', 'cartwheel', 'eat', 'golf', 'hit', 'laugh', 'shoot ball', 'shoot gun', 'smile', 'somersault', 'swing baseball', 'fencing', 'kick', 'kick ball', 'pick', 'pour', 'pushup', 'run', 'sit', 'smoke', 'talk']
hmdb_cls = [f"a photo of action about {i}" for i in hmdb_c]

ucf_c=['apply eye makeup', 'archery', 'baby crawling', 'balance beam', 'band marching', 'baseball pitch', 'basketball', 'basketball dunk', 'bench press', 'biking', 'billiards', 'blow dry hair', 'body weight squats', 'bowling', 'boxing punching bag', 'boxing speed bag', 'breast stroke', 'brushing teeth', 'cricket bowling', 'drumming', 'fencing', 'field hockey penalty', 'frisbee catch', 'front crawl', 'haircut', 'hammering', 'head massage', 'hula hoop', 'javelin throw', 'juggling balls', 'jumping jack', 'kayaking', 'knitting', 'long jump', 'lunges', 'military parade', 'mixing', 'mopping floor', 'nunchucks', 'parallel bars', 'pizza tossing', 'playing cello', 'playing dhol', 'playing flute', 'playing piano', 'playing sitar', 'playing tabla', 'playing violin', 'pole vault', 'pull ups', 'push ups', 'rafting', 'rope climbing', 'rowing', 'shaving beard', 'skijet', 'soccer juggling', 'soccer penalty', 'sumo wrestling', 'swing', 'table tennis shot', 'tai chi', 'throw discus', 'trampoline jumping', 'typing', 'uneven bars', 'walking with dog', 'wall pushups', 'writing on board', 'yo yo', 'apply lipstick', 'cricket shot', 'hammer throw', 'handstand pushups', 'high jump', 'horse riding', 'playing daf', 'playing guitar', 'shot put', 'skate boarding', 'blowing candles', 'clean and jerk', 'cliff diving', 'cutting in kitchen', 'diving', 'floor gymnastics', 'golf swing', 'handstand walking', 'horse race', 'ice dancing', 'jump rope', 'pommel horse', 'punch', 'rock climbing indoor', 'salsa spin', 'skiing', 'sky diving', 'still rings', 'surfing', 'tennis swing', 'volleyball spiking']
ucf_cls = [f"a photo of action about {i}" for i in ucf_c]
test_c=['high jump','pole vault']
test_cls = [f"a photo of action about {i}" for i in ucf_c]

smsm_c = ['pouring something into something', 'poking a stack of something without the stack collapsing', 'pretending to poke something', 'lifting up one end of something without letting it drop down', 'moving part of something', 'moving something and something away from each other', 'removing something, revealing something behind', 'plugging something into something', 'tipping something with something in it over, so something in it falls out', 'stacking number of something', "putting something onto a slanted surface but it doesn't glide down", 'moving something across a surface until it falls down', 'throwing something in the air and catching it', 'putting something that cannot actually stand upright upright on the table, so it falls on its side', 'holding something next to something', 'pretending to put something underneath something', "poking something so lightly that it doesn't or almost doesn't move", 'approaching something with your camera', 'poking something so that it spins around', 'pushing something so that it falls off the table', 'spilling something next to something', 'pretending or trying and failing to twist something', 'pulling two ends of something so that it separates into two pieces', 'lifting up one end of something, then letting it drop down', "tilting something with something on it slightly so it doesn't fall down", 'spreading something onto something', 'touching (without moving) part of something', 'turning the camera left while filming something', 'pushing something so that it slightly moves', 'uncovering something', 'moving something across a surface without it falling down', 'putting something behind something', 'attaching something to something', 'pulling something onto something', 'burying something in something', 'putting number of something onto something', 'letting something roll along a flat surface', 'bending something until it breaks', 'showing something behind something', 'pretending to open something without actually opening it', 'pretending to put something onto something', 'moving away from something with your camera', 'wiping something off of something', 'pretending to spread air onto something', 'holding something over something', 'pretending or failing to wipe something off of something', 'pretending to put something on a surface', 'moving something and something so they collide with each other', 'pretending to turn something upside down', 'showing something to the camera', 'dropping something onto something', "pushing something so that it almost falls off but doesn't", 'piling something up', 'taking one of many similar things on the table', 'putting something in front of something', 'laying something on the table on its side, not upright', 'lifting a surface with something on it until it starts sliding down', 'poking something so it slightly moves', 'putting something into something', 'pulling something from right to left', 'showing that something is empty', 'spilling something behind something', 'letting something roll down a slanted surface', 'holding something behind something', 'lifting something up completely without letting it drop down', 'pouring something into something until it overflows', 'putting something, something and something on the table', 'trying to bend something unbendable so nothing happens', 'pouring something out of something', 'throwing something onto a surface', 'putting something onto something else that cannot support it so it falls down', 'pretending to pour something out of something, but something is empty', 'pulling something out of something', 'holding something in front of something', 'tilting something with something on it until it falls off', 'moving something away from the camera', 'twisting (wringing) something wet until water comes out', 'poking a hole into something soft', 'pretending to take something from somewhere', 'putting something upright on the table', 'poking a hole into some substance', 'rolling something on a flat surface', 'poking a stack of something so the stack collapses', 'twisting something', 'something falling like a feather or paper', 'putting something on the edge of something so it is not supported and falls down', 'pushing something off of something', 'dropping something into something', 'letting something roll up a slanted surface, so it rolls back down', 'pushing something with something', 'opening something', 'putting something on a surface', 'taking something out of something', 'spinning something that quickly stops spinning', 'unfolding something', 'moving something towards the camera', 'putting something next to something', 'scooping something up with something', 'squeezing something', 'failing to put something into something because something does not fit']
smsm_cls = [f"a photo of action about {i}" for i in smsm_c]

kinetics_c = ['air drumming', 'arm wrestling', 'beatboxing', 'biking through snow', 'blowing glass', 'blowing out candles', 'bowling', 'breakdancing', 'bungee jumping', 'catching or throwing baseball', 'cheerleading', 'cleaning floor', 'contact juggling', 'cooking chicken', 'country line dancing', 'curling hair', 'deadlifting', 'doing nails', 'dribbling basketball', 'driving tractor', 'drop kicking', 'dying hair', 'eating burger', 'feeding birds', 'giving or receiving award', 'hopscotch', 'jetskiing', 'jumping into pool', 'laughing', 'making snowman', 'massaging back', 'mowing lawn', 'opening bottle', 'playing accordion', 'playing badminton', 'playing basketball', 'playing didgeridoo', 'playing ice hockey', 'playing keyboard', 'playing ukulele', 'playing xylophone', 'presenting weather forecast', 'punching bag', 'pushing cart', 'reading book', 'riding unicycle', 'shaking head', 'sharpening pencil', 'shaving head', 'shot put', 'shuffling cards', 'slacklining', 'sled dog racing', 'snowboarding', 'somersaulting', 'squat', 'surfing crowd', 'trapezing', 'using computer', 'washing dishes', 'washing hands', 'water skiing', 'waxing legs', 'weaving basket', 'baking cookies', 'crossing river', 'dunking basketball', 'feeding fish', 'flying kite', 'high kick', 'javelin throw', 'playing trombone', 'scuba diving', 'skateboarding', 'ski jumping', 'trimming or shaving beard', 'blasting sand', 'busking', 'cutting watermelon', 'dancing ballet', 'dancing charleston', 'dancing macarena', 'diving cliff', 'filling eyebrows', 'folding paper', 'hula hooping', 'hurling (sport)', 'ice skating', 'paragliding', 'playing drums', 'playing monopoly', 'playing trumpet', 'pushing car', 'riding elephant', 'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe', 'unboxing']
kinetics_cls = [f"a photo of action about {i}" for i in kinetics_c]
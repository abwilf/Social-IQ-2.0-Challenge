# Don't bother reading this!  Just utility functions.
# pip install pandas numpy requests tqdm h5py pyyaml
import shutil, os, pathlib, pickle, sys, math, importlib, json.tool, argparse, requests, atexit, builtins, itertools, hashlib, tarfile, copy, random
import pandas as pd
import numpy as np
from glob import glob
from os.path import exists, isdir
from tqdm import tqdm
from itertools import product
from datetime import datetime
import time
from collections import defaultdict
# from sklearn.utils import class_weight
import pprint
import re
import yaml
from io import StringIO 

np.cat = np.concatenate

def set_seed(seed):
    if seed in [-1,0]:
        seed = random.randint(0,100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cat_errs():
    a = os.popen('squeue | grep awilf').read()
    print('\n'.join(lmap(lambda elt: f'cat logs/{elt.strip().split(" ")[0]}.err', a.strip().split('\n'))))

def process_defaults(defaults=[], parser_in=None, str_lists=None, str_lists_type=None):
    '''
    defaults = [
        ('--hi', int, 1)
    ]
    def main():
        global args
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_type", type=str, required=True, help="The model architecture",)
        
        # pass parser_in if you want to add some other arguments with argparse in a different form, else leave parser_in=None and it'll create from defaults alone
        args = process_defaults(defaults, parser_in=parser) 

    str_lists is if you want to process some arguments into a list passed in as a string
    e.g.
        args = process_defaults(defaults, parser_in=parser, str_lists=['alpha', 'mu', 'sig'], str_lists_type=float)
    
    if str_lists_type is a list, will treat each arg differently; if single value, will treat all of them as same type
    '''
    parser = parser_in if parser_in is not None else argparse.ArgumentParser()
    for default in defaults:
        _name, _type, _default = default
        parser.add_argument(_name, type=_type, default=_default)
    args = parser.parse_args()

    if str_lists is not None:
        if not isinstance(str_lists_type, list):
            str_lists_type = [str_lists_type for _ in range(len(str_lists))]
        for arg,_type in zip(str_lists, str_lists_type):
            setattr(args, arg, lmap(lambda elt: _type(elt), lfilter(lambda elt: elt != '', getattr(args, arg).split(','))))

    return args

def tens(x):
    # x must be a list of values or a single value
    if isinstance(x, list):
        return torch.tensor(x)
    else:
        return torch.tensor([x])


def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def write_tar(filename, file_list):
    rm(filename)
    file_obj= tarfile.open(filename,"w")
     
    for elt in file_list:
        file_obj.add(elt)
     
    file_obj.close()

def get_args(defaults,gc,sg_path='/work/awilf/Standard-Grid'):
    import sys; sys.path.append(sg_path); import standard_grid
    # takes defaults, initializes argparse, modifies gc to contain all args
    # e.g. defaults = [
    #     ('--out_dir', str, 'fakeresults/hi1'),
    #     ('--hp1', int, 0)
    # ]

    parser = standard_grid.ArgParser()
    for arg in defaults:
        parser.register_parameter(*arg)
    args = parser.compile_argparse()

    for arg, val in args.__dict__.items():
        gc[arg] = val

    return gc

def load_pknone(path, func, args):
    # load pk from path, if none execute func with args, save result in path and return
    pk = load_pk(path)
    if pk is None:
        pk = func(*args)
        save_pk(path, pk)
    return pk
    

def flatten(x):
    '''x is list of lists'''
    merged = list(itertools.chain.from_iterable(x))
    return merged

def pstring(elt):
    import pprint
    return pprint.pformat(elt,indent=4)

pp = pprint.PrettyPrinter(indent=4).pprint

def join(*args):
    '''os.path.join but turns all args into strings'''
    new_args = [str(elt) for elt in args]
    return os.path.join(*new_args)

def get_sample_weight(labels,class_weights=None):
    labels = labels.astype('int32')
    if class_weights is None:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels).astype('int32'), y=labels)
    sample_weight = lvmap(lambda elt: class_weights[elt], labels)
    return sample_weight

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features 

def pk_to_amir_csd(pk, path,metadata=None):
    '''
    pk of form {
        'k1': {
            'features': np array
            'intervals': np array
        }
    }

    writes this pk to path, which should be of form /a/b/hi.csd, where hi will be the rootname
    '''
    import h5py
    rm(path)
    with h5py.File(path, 'w') as file_handle:
        rootname = path.split('/')[-1].replace('.csd', '')
        root_handle = file_handle.create_group(rootname)
        data_handle = root_handle.create_group('data')
        metadata_handle = root_handle.create_group('metadata')
        # metadata_handle['root name'] = rootname

        for vid in pk:
            vid_handle = data_handle.create_group(vid)

            vid_handle.create_dataset('features',data=pk[vid]['features'])
            vid_handle.create_dataset('intervals',data=pk[vid]['intervals'])

        metadata_handle.create_dataset('root name', data=np.arange(2))
        file_handle.close()

def csd_to_pk(ds, path=None):
    '''ds is in h5py format, but just looking at the different keys in this group, each of which has features and intervals datasets'''
    new_text = {}
    for k in ds.keys():
        new_text[k] = {
            'features': ar(ds[k]['features']),
            'intervals': ar(ds[k]['intervals']),
        }
    if path is not None:
        save_pk(path, new_text)
    return new_text
    
def amir_csd_to_pk(path):
    import h5py
    with h5py.File(path, 'r') as f:
        rootname = path.split('/')[-1].replace('.csd', '')
        ds = f[rootname]['data']
        pk = csd_to_pk(ds)
    return pk


def lvmap(f, arr, axis=None):
    if axis is None:
        f = np.vectorize(f)
        return f(arr)
    else:
        return np.apply_along_axis(f,axis=axis,arr=arr)
    
def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        # update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        # update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)


def send_email(subject='Mailgun', text='Hello', to_addr=None, secrets_path='/work/awilf/utils/mailgun_secrets.json'):
    secrets = load_json(secrets_path)
    return requests.post(
		secrets['url'],
		auth=("api", secrets['api_key']),
		data={"from": secrets['from_addr'],
			"to": [secrets['to_addr'] if to_addr is None else to_addr],
			"subject": subject,
			"text": text}
    )

# from datetime import datetime
class Runtime():
    def __init__(self):
        self.start_time = datetime.now()
    def get(self):
        end_time = datetime.now()
        sec = (end_time - self.start_time).seconds
        days = int(sec/(3600*24))
        hrs = int(sec/3600)
        mins = int((sec % 3600)/60)
        
        days_str = f'{days} days, ' if days > 0 else ''
        hrs_str = f'{hrs} hrs, ' if hrs > 0 else ''
        # print(f'\nEnd time: {end_time}')
        print(f'Runtime: {days_str}{hrs_str}{mins} mins')

def update_gpu_log(gpu_id, status):
    if gpu_id is None:
        return
    gpu_log = load_json(gpu_log_path)
    gpu_log[str(gpu_id)] = status
    save_json(gpu_log_path, gpu_log)

def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        if not test:
            send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)

def obj_to_grid(a):
    '''get all objects corresponding to hyperparamter grid search
    a = {'b': [1,2], 'c': [3,4], 'd': 5}
    ->
    [{'b': 1, 'c': 3, 'd': 5}, {'b': 1, 'c': 4, 'd': 5}, {'b': 2, 'c': 3, 'd': 5}, {'b': 2, 'c': 4, 'd': 5}]
    '''

    for k,v in list(a.items()):
        if type(v) != list:
            a[k] = [v]

    to_ret = []
    for values in list(product(*list(a.values()))):
        to_ret.append({k:v for k,v in zip(a.keys(), values)})
    return to_ret

def ar(a):
    return np.array(a)

def cp(src, dst):
    shutil.copy(src, dst)

def cpr(src,dst): 
    shutil.copytree(src,dst)

def rm(filepath):
    if exists(filepath):
        os.remove(filepath)

def rmrf(dir_path):
    if exists(dir_path):
        print(f'Removing {dir_path}')
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            os.remove(dir_path)
rmtree = rmrf

def read_txt(filename):
    with open(filename, 'r') as f:
        s = f.read()
    return s

def write_txt(filename, s):
    with open(filename, 'w') as f:
        f.write(s)
        
def remove_inf(x):
    x[x==-np.inf] = 0
    x[x==np.inf] = 0
    return np.array(x, dtype='float32')

def npr(x, decimals=4):
    '''Round'''
    return np.round(x, decimals=decimals)

def nprs(x, decimals=2, scale=100):
    '''Round & scale'''
    return np.round(x*scale, decimals=decimals)

def int_to_str(*keys):
    return [list(map(lambda elt: str(elt), key)) for key in keys]

def rsp(elt): # get last element in path; changed impl so that /a/b/ gives you b, not ''
    # return elt.rsplit('/',1)[-1]
    return lfilter(lambda elt: elt != '', elt.rsplit('/'))[-1]


def file_id(elt, ext): # get file_id of some /path/to/obj.ext
    return rsp(elt).split(ext)[0]
    
def rm_mkdirp(dir_path, overwrite, quiet=False):
    if os.path.isdir(dir_path):
        if overwrite:
            if not quiet:
                print('Removing ' + dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)

        else:
            print('Directory ' + dir_path + ' exists and overwrite flag not set to true.  Exiting.')
            exit(1)
    if not quiet:
        print('Creating ' + dir_path)
    pathlib.Path(dir_path).mkdir(parents=True)

def lists_to_2d_arr(list_in, max_len=None):
    '''2d list in, but where sub lists may have differing lengths, one big padded 2d arr out'''
    max_len = max([len(elt) for elt in list_in]) if max_len is None else max_len
    new_arr = np.zeros((len(list_in), max_len))
    for i,elt in enumerate(list_in):
        if len(elt) < max_len:
            new_arr[i,:len(elt)] = elt
        else:
            new_arr[i,:] = elt[:max_len]
    return new_arr


def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rmfile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    
def rglob(dir_path, pattern):
    return list(map(lambda elt: str(elt), pathlib.Path(dir_path).rglob(pattern)))

def move_matching_files(dir_path, pattern, new_dir, overwrite):
    rm_mkdirp(new_dir, True, overwrite)
    for elt in rglob(dir_path, pattern):
        shutil.move(elt, new_dir)
    
def mv(a,b):
    shutil.move(a,b)

def df_sample():
    return pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns=['one', 'two', 'three'])

def cossim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def subset(a, b):
    return np.min([elt in b for elt in a]) > 0

def subsets_equal(a,b):
    if len(a) == 0:
        return len(b) == 0
    if len(b) == 0:
        return len(a) == 0
    return subset(a,b) and subset(b,a)

subsets_eq = subsets_equal

def pairs_to_arr(pairs):
    # pairs of idxs to an array in graph nn form: e.g. pairs = [(0,0), (0,1)] -> [[0,0], [0,1]]
    return ar(lzip(*pairs)).reshape(2,-1)

def arr_to_pairs(arr):
    return lzip(ar(arr)[0,:], ar(arr)[1,:])

def pointers_eq(conn1, conn2):
    # conn1 and conn2 are of shape (2,x), where each pair is a start, end coord: for graph NN testing
    # conn1 = ar([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    # conn2 = ar([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    conn1 = ar(conn1)
    conn2 = ar(conn2)
    conn1set = arr_to_pairs(conn1)
    conn2set = arr_to_pairs(conn2)
    return subsets_equal(conn1set, conn2set)

def pointers_diff(conn1, conn2):
    conn1 = ar(conn1)
    conn2 = ar(conn2)
    conn1set = arr_to_pairs(conn1)
    conn2set = arr_to_pairs(conn2)

    return [elt for elt in conn1set if elt not in conn2set], [elt for elt in conn2set if elt not in conn1set]

def dict_at(d):
    k = lkeys(d)[0]
    return k, d[k]

def dict_val(d):
    return dict_at(d)[1]

def list_gpus():
    return tf.config.experimental.list_physical_devices('GPU')

def sh_to_launch(a, launch_path='/work/awilf/MTAG/.vscode/launch.json'):
    '''
    e.g. 
    a =
    # BEST FACTORIZED
    python main.py \
    --bs 10 \
    --drop_het 0 \
    ...

    or a = 'run_factorized.sh'
    '''
    if exists(a):
        a = read_txt(a).replace('\\','').replace('\n','')

    prog, args = a.strip().replace('#','').split('.py')
    prog = prog.split("python")[1].strip() + '.py'
    args = [elt for elt in args.split(' ') if elt != '']

    pk = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                # "env": { "CUDA_LAUNCH_BLOCKING": "1" },
                "program": f"{prog}",
                "console": "integratedTerminal",
                "justMyCode": False,
                "args": args
            }
        ]
    }
    save_json(launch_path, pk)
    return pk

def gdown_str(url):
    return f'https://drive.google.com/uc?id={url.split("/")[-2]}'
    
def launch_to_sh(sh_path):
    a = load_json('.vscode/launch.json')
    config = a['configurations'][0]
    program_type = config['type']
    program_name = config['program']
    args = config['args']
    s = f'{program_type} {program_name} \\\n'  + ' \\\n'.join([' '.join(elt) for elt in lzip(args[:-1:2], args[1::2])])
    write_txt(sh_path, s)
    return s

def save_pk(file_stub, pk, protocol=4):
    filename = file_stub if ('.pk' in file_stub or '.pickle' in file_stub or 'pkl' in file_stub) else f'{file_stub}.pk'
    rmfile(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)
    
def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        return load_pk_old(filename)

def load_pk_old(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

# def ds(d, *keys):
#     '''Destructure dict. e.g.
#     a = {'hey': 1, 'you': 2}
#     hey, you = ds(a, 'hey', 'you')
#     '''
#     return [ d[k] if k in d else None for k in keys ]

def get_dir(path, silent=True):
    if '.' not in path and not silent:
        print(f'NOTE: {path} is not a file, creating dir with just {path}')
    else:
        path = '/'.join(path.split('/')[:-1])
    return path

def get_ints(*keys):
    return [int(key) for key in keys]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

def load_json(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename) as json_file:
        return json.load(json_file)


def load_jsonl(path):
    if not exists(path):
        return None
    arr = []
    with open(path, 'r') as f:
        for idx, l in tqdm(enumerate(f)):
            item = json.loads(l)
            arr.append(item)
    return arr

def save_jsonl(path, obj):
    assert isinstance(obj, list)

    with open(path, 'w') as outfile:
        for entry in obj:
            json.dump(entry, outfile)
            outfile.write('\n')


def lfilter(fn, iterable):
    return list(filter(fn, iterable))

def lkeys(obj):
    return list(obj.keys())

def lvals(obj):
    return list(obj.values())

def lmap(fn, iterable):
    return list(map(fn, iterable))

def arlmap(fn, iterable):
    return ar(list(map(fn, iterable)))

def arlist(x):
    return ar(list(x))
    
def llmap(fn, iterable):
    return list(map(lambda elt: fn(elt), iterable))
    
def sort_dict(d, reverse=False):
    return {k: v for k,v in sorted(d.items(), key=lambda elt: elt[1], reverse=reverse)}

def csv_path(sym):
    return join('csvs', f'{sym}.csv')

def is_unique(a):
    return len(np.unique(a)) == len(a)

def lists_equal(a,b):
    return np.all([elt in b for elt in a]) and np.all([elt in a for elt in b])
    
def split_arr(cond, arr):
    return lfilter(cond, arr), lfilter(lambda elt: not cond(elt), arr)

def lzip(*keys):
    return list(zip(*keys))

def dilation_pad(max_len, max_dilation_rate):
    to_ret = math.ceil(max_len/max_dilation_rate)*max_dilation_rate
    assert (to_ret % max_dilation_rate) == 0
    return to_ret

def all_eq(arr):
    return (arr==arr[0]).all()

def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

def onehot_initialization(a):
    '''a is 2d with values ranging from 0 to num_labels-1.  this turns a into a 3d matrix with same first two dimensions, one hot encoded'''
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

def zero_pad_to_length(data, length):
    padAm = length - data.shape[0]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,padAm), (0,0)), 'constant')

def get_batch(arr, batch_idx, batch_size):
    return arr[batch_idx * batch_size:(batch_idx + 1) * batch_size]

def sample_batch(arrs, batch_size):
    start = np.random.randint(arrs[0].shape[0]-batch_size)
    return [arr[start:(start+batch_size)] for arr in arrs]

def shuffle_data(*arrs):
    rnd_state = np.random.get_state()
    for arr in arrs:
        np.random.shuffle(arr)
        np.random.set_state(rnd_state)

def get_class_weights(arr):
    '''pass in dummies'''
    class_weights = np.nansum(arr, axis=0)
    return np.sum(class_weights) / (class_weights*len(class_weights))

def get_class_weights_ds(arr):
    '''do not pass in dummies'''
    arr = np.stack(np.unique(np.array(arr), return_counts=True), axis=1)
    return (np.sum(arr[:,1]) - arr[:,1]) / arr[:,1]

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

'''
mapping from original preprocessed mosei data to form used in rest of .pickle files
def map_13_to_7(mode):
    a = load_pk(f'/z/abwilf/hffn/multimodal-sentiment-analysis/dataset/mosei/{mode}_3way.pickle')
    new_obj = [None]*7
    new_obj[0] = np.concatenate([a[0], a[2]])
    new_obj[1] = np.concatenate([a[1], a[3]])
    new_obj[2] = a[4]
    new_obj[3] = a[5]
    new_obj[4] = a[6]
    new_obj[5] = np.concatenate([a[7], a[8]])
    new_obj[6] = a[9]
    save_pk(f'/z/abwilf/hffn/reshaped_theirs/{mode}.pk', new_obj)
[map_13_to_7(mode) for mode in ['audio', 'text', 'video']]
'''

def rreplace(s, old, new, occurrence):
    '''replace the last occurrence # of of old characters with new characters in a str'''
    li = s.rsplit(old, occurrence)
    return new.join(li)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LD(dict):
    '''Dict mod that allows multi indexing'''
    def __init__(self, dict_in):
        self.__dict__ = dict_in
    def lcg(self,key,val=[]):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            self.__dict__[key] = val
            return self.__dict__[key]
    def __getitem__(self, key):
        if type(key) == list:
            return [self.__dict__[k] for k in key]
        else:
            return self.__dict__[key]
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    def has_key(self, k):
        return k in self.__dict__
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def pop(self, *args):
        return self.__dict__.pop(*args)
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __unicode__(self):
        return unicode(repr(self.__dict__))

def z_norm(arr):
    mean, std = arr.mean(), arr.std()
    return (arr-arr.mean()) / (arr.std() + 1e-5), mean, std

def un_z_norm(arr, mean, std):
    return arr*std + mean

# -- main wrapper for stdgrid; usage documented in main_wrapper -- #
gc = {}
def get_arguments(defaults):
    import sys; sys.path.append('/work/awilf/Standard-Grid'); import standard_grid
    parser = standard_grid.ArgParser()
    for arg in defaults:
        parser.register_parameter(*arg)

    args = parser.compile_argparse()

    global gc
    for arg, val in args.__dict__.items():
        gc[arg] = val
    return gc

def write_results(results):
    mkdirp(gc['out_dir'])
    save_json(join(gc['out_dir'], 'results.json'), results)
    write_txt(join(gc['out_dir'], 'success.txt'), '')

def main_wrapper(main_fn, defaults, results=True, runtime=True):
    '''
    defaults: array of 3-tuples defining arguments and their default values
    main_fn takes in _gc, turns it into a global variable as below, returns results dictionary


    # example main.py
    import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
    from args import defaults

    def main(_gc):
        global gc; gc = _gc

        # ... do whatever ...

        results = {
            'accuracy': [...], # or whatever you want to save
            'loss': [...],
        }
        return results

    if __name__ == '__main__':
        main_wrapper(main,defaults)


    # example args.py
        defaults = [
            ("--out_dir", str, "results/hash1"), # REQUIRED - results will go here
            ("--rationale", float, 1.2), # other arguments
            ('--debug', int, 0),
        ]
    '''

    rt = Runtime()
    global gc
    get_arguments(defaults)

    if results:
        mkdirp(gc['out_dir'])
        rm(join(gc['out_dir'], 'success.txt'))
        rm(join(gc['out_dir'], 'results.json'))
    
    res = main_fn(gc)
    if results:
        write_results(res)
    if runtime:
        rt.get()

standard_wrapper = main_wrapper
common_wrapper = main_wrapper





#### NON-STANDARD GRID (NSG) ####
def get_grid(hp):
    if 'subsets' in hp.keys():
        all_grids = []
        subsets = hp['subsets']
        del hp['subsets']

        for sub in subsets:
            sub_hp = {**hp, **sub}
            keys, vals = zip(*list(sub_hp.items()))
            grid = [{k:v for k,v in zip(keys,elt)} for elt in list(itertools.product(*vals))]
            all_grids.extend(grid)

        grid = all_grids
    else:
        keys, vals = zip(*list(hp.items()))
        grid = [{k:v for k,v in zip(keys,elt)} for elt in list(itertools.product(*vals))]
    
    return grid

def get_ops(hash_, config):
    return os.popen(f'squeue | grep {config["andrewid"]} | grep {hash_[:config["num_chars_squeue"]]}').read()

def get_num_ops(hash_, config):
    return get_ops(hash_, config).count('\n')

def get_id(path):
    return path.split('/')[-1].replace('.sh', '')

def create_dir_structure(hash_, hash_path, grid, config):
    ## Create directory structure and runscripts within results/, looks like this at the end (after err and out files have been written)
    '''
    results/cb97ca32aaf0497/
    ├── 0
    │   ├── compute-0-18-err.txt
    │   ├── compute-0-18-out.txt
    │   ├── results.json
    │   └── success.txt
    ├── 1
    │   ├── compute-0-18-err.txt
    │   ├── compute-0-18-out.txt
    │   ├── results.json
    │   └── success.txt
    ├── 2
    ...
    ├── csv_results.csv
    ├── compressed.tar
    ├── hp.json
    ├── report.json
    └── run_scripts
        ├── 0.sh
        ├── 1.sh
        ├── 2.sh
        ...
    '''
    print('Length of grid:', len(grid))
    if isdir(hash_path):
        if not config['overwrite']:
            print(f'Hash path {hash_path} exists and overwrite is not specified. Exiting now.')
            exit()
        else:
            print(f'Removing and rewriting hash path {hash_path}')
            rmrf(hash_path)

    run_scripts_dir = join(hash_path, 'run_scripts')
    mkdirp(run_scripts_dir)

    to_run = []
    for i,comb in enumerate(grid):
        out_dir = join(hash_path, str(i))
        mkdirp(out_dir)

        hp_to_add = ' '.join([f'--{k} {v}' for k,v in comb.items()]) + f' --out_dir {out_dir}'

        # modify skeleton to create final sh file: add output files and hp flags
        skel = config['skeleton'].split('\n')
        split_idx = np.max([i for (i,elt) in enumerate(skel) if '#SBATCH' in elt])
        before = skel[:split_idx+1]
        after = skel[split_idx+1:]
        middle = [
            f'#SBATCH --job-name {i}_{hash_}        # %j specifies JOB_ID',
            f'#SBATCH -o {join(out_dir,"%N-out.txt")}        # STDOUT, says which machine in case you want to exclude',
            f'#SBATCH -e {join(out_dir,"%N-err.txt")}        # STDERR',
        ]
        skel = '\n'.join([*before, *middle, *after])
        skel = skel.replace(config['command'], f"{config['command']} {hp_to_add}")
        
        run_script = join(run_scripts_dir, f'{i}.sh')
        to_run.append(run_script)
        write_txt(run_script, skel)
    
    return to_run


def scancel_all(usr='awilf'):
    s = '\n'.join(lmap(lambda elt: f'scancel {elt}', lfilter(lambda elt: elt != '', lmap(lambda elt: elt.strip().split(' ')[0], os.popen(f'squeue | grep {usr}').read().split('\n')))))
    s = s.rstrip().replace('\n', ' && ')
    print(s)

def submit_scripts(to_run, in_progress, hash_, config):
    num_sbatch_ops = get_num_ops(hash_, config)
    while num_sbatch_ops <= config['max_sbatch_ops'] and len(to_run) > 0:
        run_script = to_run.pop()
        in_progress.append(run_script)
        os.popen(f'sbatch {run_script}').read()
        num_sbatch_ops = get_num_ops(hash_, config)

def monitor(to_run, in_progress, finished, tot_num, hash_, config):
    submit_scripts(to_run, in_progress, hash_, config)
    if len(to_run) == 0 and len(in_progress)==0:
        config['gridsearch_complete'] = True
        return in_progress, finished

    sbatch_ops = get_ops(hash_, config)
    finished.extend([elt for elt in in_progress if f'{get_id(elt)}_{hash_[:config["num_chars_squeue"]]}' not in sbatch_ops])
    in_progress = [elt for elt in in_progress if f'{get_id(elt)}_{hash_[:config["num_chars_squeue"]]}' in sbatch_ops]

    num_sbatch_ops = get_num_ops(hash_, config)
    print(f'To run: {100*len(to_run) / tot_num:.1f}%\tIn progress: {100*len(in_progress)/tot_num:.1f}%\tFinished: {100*len(finished)/tot_num:.1f}%', end='\r')

    time.sleep(config['sleep_secs'])
    return in_progress, finished

def submit_monitor_sbatch(to_run, hash_, config):
    # Submit and monitor script progress (not too many at at time)
    in_progress, finished = [], []
    tot_num = len(to_run)
    config['gridsearch_complete'] = False

    print(f'\n\nhash=\'{hash_}\'\n')
    print(f'\n## Status ## \nRunning {len(to_run)} scripts total (max {config["max_sbatch_ops"]} at a time)\n')

    while not config['gridsearch_complete']:
        in_progress, finished = monitor(to_run, in_progress, finished, tot_num, hash_, config)

    print('\n\nGrid Search Complete!')

def collate_results(hash_, hash_path, grid, config):
    # Consolidate json files into a single csv
    csv_path = join(hash_path, 'csv_results.csv')
    print(f'Writing csv to \n{csv_path}\n')

    ld = {} # list of dicts
    for path in pathlib.Path(join(config["results_path"], hash_)).rglob('*.json'):
        id = int(str(path).split('/')[-2])
        hp_comb = grid[id]
        ld[id] = {**load_json(path), **{'_'+k:v for k,v in hp_comb.items()}}

    df = pd.DataFrame(ld).transpose()
    df.to_csv(csv_path)

    hp_path = join(hash_path, 'hp.json')
    save_json(hp_path, config['hp'])

def compile_error_report(hash_path, grid):
    # compile error report
    report = {
        'num_combs': len(grid),
        'num_successful': 0,
        'num_failed': 0,
        'errors': {}
    }
    for i in range(len(grid)):
        if 'success.txt' not in [elt.split('/')[-1] for elt in glob(join(hash_path, f'{i}', '*'))]:
            report['num_failed'] += 1
            report['errors'][i] = {
                'hp': grid[i],
                'err': open([elt for elt in glob(join(hash_path, f'{i}', '*')) if 'err.txt' in elt][0]).read(),
                'node': [elt for elt in glob(join(hash_path, f'{i}', '*')) if 'err.txt' in elt][0].split('-err')[0],
            }
        else:
            report['num_successful'] += 1

    if report['num_failed'] > 0:
        print(f'###\n!!! ALERT !!!\nThere were some errors.  Please check the report for a description:\n{join(hash_path, "report.json")}\n###')
    
    save_json(join(hash_path, 'report.json'), report)

def email_complete(config):
    os.popen(f'sbatch --mail-type=END --mail-user={config["mail_user"]} --wrap "{config["dummy_program"]}"')

def compress_files(hash_path, config):
    if len(config['tarfiles']) > 0:
        write_tar(join(hash_path, 'compressed.tar'), config['tarfiles'])
    

def nsg(config):
    '''
    Requires a runfile (e.g. /work/awilf/utils/run_nsg.py)
    '''
    rt = Runtime()
    hash_ = hashlib.sha1(json.dumps(config['hp'], sort_keys=True).encode('utf-8')).hexdigest()[:config['hash_len']]
    hash_path = join(config['results_path'], hash_)

    grid = get_grid(config['hp'])
    to_run = create_dir_structure(hash_, hash_path, grid, config)
    submit_monitor_sbatch(to_run, hash_, config)
    collate_results(hash_, hash_path, grid, config)
    compile_error_report(hash_path, grid)
    email_complete(config)
    compress_files(hash_path, config)

    print(f'\nhash=\'{hash_}\'\n\n')

    rt.get()
####

def read_yml(filename):
    t = read_txt(filename)
    try:
        return yaml.safe_load(t)
    except yaml.YAMLError as exc:
        print(exc)
        return None

def write_yml(filename, d):
    with open(filename, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

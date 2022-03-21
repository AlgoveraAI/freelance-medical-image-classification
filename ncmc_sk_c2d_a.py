import os
from fastai.vision.all import *
from fastai.vision.learner import cnn_learner, create_head, create_body, num_features_model, default_split, has_pool_type, apply_init, _update_first_layer
from fastai.callback.wandb import WandbCallback
from sklearn.model_selection import StratifiedKFold
from random import sample
from timm import create_model
#import wandb
import sys
import json
import time


SEED=101
random.seed(SEED)
set_seed(SEED, True)
#wandbk = 'abc032abd4aceee91f5886a4615823dec09722bd253abc'
#wandb.login(key=wandbk[3:-3])
start = time.time()

#configs for timm-based models
TimmConfig= {
    #non-transformer
    'efficientnet_b0':{'arch': 'efficientnet_b0', 'is_transformer':False},
    'efficientnet_b3':{'arch': 'efficientnet_b3', 'is_transformer':False},
    'tf_efficientnet_b0_ap':{'arch':'tf_efficientnet_b0_ap', 'is_transformer':False},
    'tf_efficientnet_b3_ap':{'arch':'tf_efficientnet_b0_ap', 'is_transformer':False},
    'tf_efficientnetv2_xl_in21ft1k':{'arch':'tf_efficientnetv2_xl_in21ft1k', 'is_transformer':False},
    'ssl_resnext50_32x4d':{'arch':'ssl_resnext50_32x4d', 'is_transformer':False},
    'ssl_resnext101_32x4d':{'arch':'ssl_resnext101_32x4d', 'is_transformer':False},

    #transformer
    'swin_base_patch4_window7_224':{'arch':'swin_base_patch4_window7_224', 'is_transformer':True, 'size':224},
    'swin_base_patch4_window7_224_in22k':{'arch':'swin_base_patch4_window7_224_in22k', 'is_transformer':True, 'size':224},
    'beit_base_patch16_224':{'arch':'beit_base_patch16_224', 'is_transformer':True, 'size':224},
    'beit_base_patch16_224_in22k':{'arch':'beit_base_patch16_224_in22k', 'is_transformer':True, 'size':224},
    'deit_base_distilled_patch16_224':{'arch':'deit_base_distilled_patch16_224', 'is_transformer':True, 'size':224},
    'deit_base_distilled_patch16_384':{'arch':'deit_base_distilled_patch16_384', 'is_transformer':True, 'size':224}
}


def get_input(
    local:bool=False, # Flag to indicate local vs C2D
):
    if local:
        print("Reading local medicaldata directory.")

        # Root directory for dataset
        filename = Path('./data')

        return filename

    dids = os.getenv('DIDS', None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    cwd = os.getcwd()
    print('cwd', cwd)

    did = dids[0]
    print(f"DID: {did}")

    filename = Path(f'/data/inputs/{did}/0')
    print(f"filename: {filename}")
    print(f"time since start {(time.time() - start):.2f}s")
    return filename


def get_label(
    fn, # Image file name
):
    # if fn.suffix == '.jpeg':
    #     l_fn = f"{str(fn).split('.jpeg')[0]}.json"
    #
    # elif fn.suffix == '.png':
    #     l_fn = f"{str(fn).split('.png')[0]}.json"
    #
    # elif fn.suffix == '.jpg':
    #     l_fn = f"{str(fn).split('.jpg')[0]}.json"
    #
    # with open(l_fn, 'r') as tmp:
    #     l = json.load(tmp)

    return str(fn).split('/')[-4]


def get_patient(
    fn, # Image file name
):
    return ' '.join(str(fn).split('/')[-3])


def get_train_test(
    df:pd.DataFrame, # Dataframe built by `get_df`
):
    ids = list(df['patient_id'].unique())
    train_ids = random.sample(ids, int(len(ids)*0.8))
    test_ids = [id_ for id_ in ids if id_ not in train_ids]

    print(f"Number of training samples: {len(train_ids)}")
    print(f"Number of valid samples: {len(test_ids)}")

    df.loc[df[df['patient_id'].isin(test_ids)].index, 'is_valid'] = True

    return df


def get_df(
    local:bool, # Flag to indicate local vs C2D
):
    print("Preparing df.")
    filename = get_input(local)

    if not local:
        data = Path('data')
        if not data.exists():
            data.mkdir()
        print(f"data folder exists: {os.path.exists(str(data))}")

        print('lets unzip and print files')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(str(data))


        image_fns = get_image_files(data/'Algorithm training')

    else:
        image_fns = get_image_files(filename)


    print(f"Printing samples of image filenames: {image_fns[:3]}")

    df = pd.DataFrame(list(image_fns), columns=['fns'])

    df['label'] = df['fns'].apply(lambda x: get_label(x))
    df['patient_id'] = df['fns'].apply(lambda x: get_patient(x))
    df['is_valid'] = False

    df = get_train_test(df)

#    print("logging df to wandb")
#    wdf = df.copy()
#    wdf['fns'] = wdf['fns'].map(str)
#    wta = wandb.Artifact('dataframe', type='dataset')
#    wt = wandb.Table(dataframe=wdf)
#    wta.add(wt, name='dataframe')
#    wandb.run.log_artifact(wta)
    print(df.head(30))
    print(f"time since start {(time.time() - start):.2f}s")
    return df


def setup_dataloaders(
    df:pd.DataFrame, # Dataframe built by `get_df`
    bs:int, # Batch_size
    size:int=512, # Resize size
    augs:list=None, # List of fastai augmentations
):
    print("Setting up dls")
    if not augs:
        augs = [Brightness(),
                Contrast(),
                Hue(),
                Saturation(),
                DeterministicDihedral(),
                Hue(),
                Saturation(),
                RandomErasing(max_count=3)]

    db = DataBlock(blocks=(ImageBlock, CategoryBlock),
                get_x=ColReader('fns'),
                get_y=ColReader('label'),
                splitter=ColSplitter(),
                item_tfms=Resize(size),
                batch_tfms=setup_aug_tfms(augs)
                )
    dls = db.dataloaders(df, bs=bs)

    return dls


def get_timm_model(
    arch:str, # Architecture of the model to use
    transformer:bool=None, # Flag to indicate if its a transformer-based model
    pretrained:bool=True, # Flag to use pretrained backbone
    cut=None,
    n_in:int=3, # Number of input channels, RGB is 3
):
    "Creates a body from any model in the `timm` library."
    if not transformer:
        body = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
        _update_first_layer(body, n_in, pretrained)
        if cut is None:
            ll = list(enumerate(body.children()))
            cut = next(i for i,o in reversed(ll) if has_pool_type(o))
        body =  nn.Sequential(*list(body.children())[:cut])
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, 2)
        model = nn.Sequential(body, head)
        apply_init(model[1], nn.init.kaiming_normal_)

        return model

    else:
        return create_model(arch,
                            pretrained=pretrained,
                            num_classes=2)


def get_learner_lr(
    dls,
    model, # Model arch to use
    timm:bool=False, # True if using timm model
    pretrained:bool=True, # Use pretrained backbone
):

    print("Setting up learner.")

    if not timm:
        learner = cnn_learner(
                              dls,
                              model,
                              pretrained=pretrained,
                              metrics=accuracy
                          )

    else:
        model_config = TimmConfig[model]
        model = get_timm_model(model_config['arch'],
                               model_config['is_transformer'],
                               pretrained,
                               )
        learner = Learner(
            dls,
            model,
            metrics=accuracy
        )

#    v = learner.lr_find()
#    lr = v[0]

    return learner

def setup_train(
    local:bool, # Flag to indicate local vs C2D
    bs:int, # Batch_size to use
    epochs:int, # Number of epochs to trian after unfreeze
    freeze_epochs:int, # Number of epochs to trian before unfreeze
    lr:float, # Learning rate
    model, # Model to use
    timm:bool, # True if using timm-based model
    pretrained:bool, # Use pretrianed backbone
    run_name:str, # Name of the current experiment
):

    df = get_df(local)

    try:
        size = TimmConfig[model]['size']
    except:
        size=256

    dls = setup_dataloaders(df, bs, size)

    learner = get_learner_lr(
                      dls=dls,
                      model=model,
                      timm=timm,
                      pretrained=pretrained)

    # if not local:
    #     fname = f"outputs/{run_name}"
    #
    #     if not Path("outputs/models").exists():
    #         Path("outputs/models").mkdir()
    print(f"time since start {(time.time() - start):.2f}s")
    sbm = SaveModelCallback(fname=run_name)

    learner.freeze()
    learner.fit_one_cycle(freeze_epochs, lr_max=lr, cbs=[GradientAccumulation(16),
                                                         GradientClip(),
                                                         #WandbCallback(log_preds=False),
                                                         sbm,
                                                         TimeCallback()])

    learner.unfreeze()
    learner.fit_one_cycle(epochs, lr_max=lr, cbs=[GradientAccumulation(16),
                                                  GradientClip(),
                                                  #WandbCallback(log_preds=False),
                                                  sbm,
                                                  TimeCallback()])

    preds, targs = learner.get_preds(dl=learner.dls.valid)
    preds = torch.argmax(preds, 1).numpy()
    targs = targs.numpy()
    print(f"valid set accuracy: {sum(preds==targs)/len(targs)}")
    # cm = wandb.plot.confusion_matrix(
    #     y_true=targs,
    #     preds=preds,
    #     class_names=list(learner.dls.vocab))

    # wandb.log({"conf_mat": cm})
    for root, dirs, files in os.walk('.'):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                fn = os.path.join(root,file)
                if fn.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                    fns.append(fn)
                print(len(path) * '---', file)

    return learner


def run(
    local:bool=False
):
    config = {
        'local':local, # Flag to indicate local vs C2D
        'bs': 8,
        'epochs':25,
        'freeze_epochs':2,
        'lr':1e-3,
        'model':resnet34, #if fastai pass fastai callable function; if timm pass arch name
        'timm':False,
        'pretrained':True,
    }

    model_name = config['model'] if isinstance(config['model'], str) else config['model'].__name__
    run_name = f"{model_name}_{config['freeze_epochs']}_{config['epochs']}"

    config['run_name'] = run_name

    if local:
      print(f"You are in local env")
    if not local:
      print(f"You are in C2D")

#    wandb.init(project="algovera_ncight_kneeshoulder",
#            name=run_name)

    learner = setup_train(
                  local=config['local'],
                  bs=config['bs'],
                  epochs=config['epochs'],
                  freeze_epochs=config['freeze_epochs'],
                  lr=config['lr'],
                  model=config['model'],
                  timm=config['timm'],
                  pretrained=config['pretrained'],
                  run_name=config['run_name']
                  )

    return learner


class TimeCallback(Callback):
    def before_train(self):
        print(f"time since start {time.time() - start:.2f}s")
    def before_epoch(self):
        print(f"time since start {time.time() - start:.2f}s")
    def before_batch(self):
        print(f"time since start {time.time() - start:.2f}s")
    def before_step(self):
        print(f"time since start {time.time() - start:.2f}s")
    def after_step(self):
        print(f"time since start {time.time() - start:.2f}s")


if __name__ == "__main__":
    print(f"Is cuda available: {torch.cuda.is_available()}")

    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run(local)

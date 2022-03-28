import os
from fastai.vision.all import *

#prepare model path
model0 = "wget https://www.dropbox.com/s/3sod2uk49a4hror/resnet34_0.pkl?dl=1"
model1 = "wget https://www.dropbox.com/s/dxoucxlnn62hif5/resnet34_1.pkl?dl=1"
model2 = "wget https://www.dropbox.com/s/cbfioaw5mj0fg2w/resnet34_2.pkl?dl=1"

os.system(model0)
os.system(model1)
os.system(model2)

model_paths = ['resnet34_0.pkl?dl=1', 'resnet34_1.pkl?dl=1', 'resnet34_0.pkl?dl=1']

#prepare data path
dids = os.getenv('DIDS', None)
dids = json.loads(dids)
did = dids[0]
data_path = Path(f'/data/inputs/{did}/0')

#unzip data
with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(str('.'))

#prepare df
def get_label(
    fn, # Image file name
):
    return str(fn).split('/')[-4]

fns = get_image_files(Path("Dr Sample 3"))
df = pd.DataFrame(list(fns), columns=['fns'])
df['label'] = df.fns.apply(lambda x: get_label(x))

#load model n make preds
preds = []
for path in model_paths:
    learn = load_learner(path)
    test_dl = learn.dls.test_dl(df)
    preds.append(learn.get_preds(dl=test_dl)[0].cpu().numpy())

# average over the three models
final_pred = np.stack(preds).mean(0)

#prepare output
df = pd.concat([df, pd.DataFrame(final_pred, columns=['knee-score', 'shoulder-score'])], 1)
df['prediction'] = df['knee-score'] > df['shoulder-score']
df['prediction'] = df['prediction'].apply(lambda x: "Knee" if x is True else "Shoulder")

df.to_csv("data/outputs/final_df.csv")
print(df)

#results figure
fig, axes = plt.subplots(10, 2, figsize=(15,30))
for i, row in df.iterrows():
    axes.flatten()[i].imshow(PILImage.create(row.fns))
    axes.flatten()[i].axis("off")
    axes.flatten()[i].set_title(f"{row.label} | {row.prediction}")
    plt.tight_layout()


fig.savefig("data/outputs/preds.jpeg")

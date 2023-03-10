import os
import wave

import librosa
import pandas
import requests
import soundfile as sf
import tqdm

cwd = "yuyi"
dataset_folder = "dataset/shejiao"

audio_folder = os.path.join(dataset_folder, "audio")
os.makedirs(audio_folder, exist_ok=True)


def download(url):
    audio_file = os.path.join(audio_folder, url.split("/")[-1])
    if not os.path.exists(audio_file):
        response = requests.get(url, allow_redirects=True)
        open(audio_file, "wb").write(response.content)

    # get sample rate of audio file
    with wave.open(audio_file, "rb") as f:
        rate = f.getframerate()
    if rate != 16000:
        assert rate == 8000
        y, sr = librosa.load(audio_file, sr=8000)
        # resample wav file from 8k to 16k
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(audio_file, y, 16000)
    return audio_file


sheet_names = ["声纹", "报告数据", "报告数据2"]

rows = []
for sheet_name in sheet_names:
    data = pandas.read_excel(
        os.path.join(cwd, "data.xlsx"), header=None, sheet_name=sheet_name
    )
    for index, row in tqdm.tqdm(data.iterrows()):
        if not isinstance(row[0], str):
            continue
        try:
            audio_file = download(row[1])
        except wave.Error:
            print("error", row[1])
            continue
        rows.append({"audio": audio_file, "name": row[0]})

output_dataframe = pandas.DataFrame(rows, columns=["audio", "name"])
# sort by name
output_dataframe = output_dataframe.sort_values(by="name")
# filter name with only one sample
output_dataframe = output_dataframe.groupby("name").filter(lambda x: len(x) > 1)
# replace name with unique id starting from 10000
output_dataframe["name"] = output_dataframe["name"].astype("category")
output_dataframe["name"] = output_dataframe["name"].cat.codes + 3242

# sample without overlap
#########################################################################
# all_names = output_dataframe["name"].unique()
# # random choice 10% name as test set, and the rest as train set
# train_names = numpy.random.choice(
#     all_names, size=int(len(all_names) * 0.9), replace=False
# )
# test_names = numpy.setdiff1d(all_names, train_names)

# train_dataframe = output_dataframe[output_dataframe["name"].isin(train_names)]
# test_dataframe = output_dataframe[output_dataframe["name"].isin(test_names)]
#########################################################################

# sample with overlap
#########################################################################
# train_dataframe = output_dataframe.sample(frac=0.9, random_state=0)
# test_dataframe = output_dataframe.drop(train_dataframe.index)
#########################################################################

# print total number of names
# print("train names: ", len(train_dataframe["name"].unique()))
# print("test names: ", len(test_dataframe["name"].unique()))

# train_dataframe.to_csv(
#     os.path.join(dataset_folder, "train_list.txt"), index=False, header=False, sep="\t"
# )
# test_dataframe.to_csv(
#     os.path.join(dataset_folder, "test_list.txt"), index=False, header=False, sep="\t"
# )
print("train names: ", len(output_dataframe["name"].unique()))
output_dataframe.to_csv(
    os.path.join(dataset_folder, "train_list.txt"), index=False, header=False, sep="\t"
)

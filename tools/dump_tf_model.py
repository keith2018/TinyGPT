import numpy as np
import tensorflow as tf
import json
import os
import re

model_dir = "models/gpt2"
ret_model_path = "model_file.data"
ret_index_path = "model_index.json"

hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
ret_json = {}
ret_file_pos = 0


def set_in_nested_dict(d, keys, val):
    if not keys:
        return val
    if keys[0] not in d:
        d[keys[0]] = {}
    d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
    return d


model_index = {"blocks": [{} for _ in range(hparams["n_layer"])]}
with open(ret_model_path, "wb") as file:
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        arr = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        print(f"{name}: {arr.shape}: {arr.size}")
        name = name[len("model/"):]
        dict = {'pos': ret_file_pos, 'size': arr.size, 'shape': arr.shape}
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(model_index["blocks"][n], sub_name.split("/"), dict)
        else:
            set_in_nested_dict(model_index, name.split("/"), dict)
        ret_file_pos = ret_file_pos + 4 * arr.size
        arr.tofile(file)

file_size = os.path.getsize(ret_model_path)

if file_size == ret_file_pos:
    print(f"file save ok, total len: {ret_file_pos}")
else:
    print("file save error!")
    exit(-1)

ret_json['file_path'] = ret_model_path
ret_json['file_size'] = file_size
ret_json['model_index'] = model_index

with open(ret_index_path, "w") as f:
    json.dump(ret_json, f, indent=4)

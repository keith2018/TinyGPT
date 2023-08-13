import requests
from tqdm import tqdm


def download(url: str, fname: str, chunk_size=4096):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=4096,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    url = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001"
    fname = "./assets/gpt2/model_file.data"
    download(url, fname)

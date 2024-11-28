import requests
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import json
from time import sleep
import numpy as np

api_keys = [
    "onJpy5ejf0yRmAq6TMHJ4NpXs3KlZDNCAN6H8iNJSSKvFcMJtH66QMu4",
    "nOXbp0Y2IHQozF18bck4cdOXjrQYn66MnvME35AW9f8OuSitYWXnS07k",
    "vI4X9GjOUUCdwcGJmbqNBxHJn2zViPEDDknawZOFQq8z4MYEnv0renFm",
]
api_key_index = 0

API_KEY = api_keys[api_key_index]
PEXEL_BUCKET = "/mnt/disks/pexel-bucket"
DOWNLOAD_ROOT = os.path.join(PEXEL_BUCKET, "images")
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)


headers = {
    "Authorization": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

}

with open(os.path.join(PEXEL_BUCKET,'class_labels.json'), 'r') as file:
    class_labels = json.load(file)
CLASS_LABEL = {int(k): v for k, v in class_labels.items()}
del class_labels

def count_downloaded_images(query_id):
    query_num_json = os.path.join(PEXEL_BUCKET, "query.json")
    with open(query_num_json, 'r') as file:
        query_num = json.load(file)

    download_dir = os.path.join(DOWNLOAD_ROOT, f'{query_id:04d}')
    already_downloaded = len(glob(os.path.join(download_dir, '*.jpg')))
    print(f'{query_id}\t{already_downloaded} images already downloaded for {CLASS_LABEL[query_id]}, inside {DOWNLOAD_ROOT}/{query_id:04d}')
    
    if str(query_id) in query_num:
        print(query_num[str(query_id)])
    print()

def download_image_by_id(photo_id):
    download_dir = os.path.join(PEXEL_BUCKET, 'images')
    image_path = os.path.join(download_dir, f'{photo_id:08d}.jpg')
    if os.path.exists(image_path):
        print(f'{image_path} already exists!')
        return
    
    id_metadata_json = os.path.join(PEXEL_BUCKET, "metadata.json")
    with open(id_metadata_json, 'r') as file:
        id_metadata = json.load(file)
    assert str(photo_id) in id_metadata, f'{photo_id} not found in metadata.json!'
    image_url = id_metadata[str(photo_id)]['src']['original']

    image_response = requests.get(image_url, headers=headers)
    assert image_response.status_code == 200
    with open(image_path, 'wb') as f:
        f.write(image_response.content)
    
    assert os.path.exists(image_path)
    print(f'Downloaded {image_path}')
    return


# Function to download images from Pexels API
def download_images_by_query(query_id, total_images=1000, per_page=80):
    
    id_metadata_json = os.path.join(PEXEL_BUCKET, "metadata.json")
    # id_metadata_json = os.path.join(PEXEL_BUCKET, "images_init/metadata_dogcat.json")
    query_num_json = os.path.join(PEXEL_BUCKET, "query.json")
    with open(id_metadata_json, 'r') as file:
        id_metadata = json.load(file)
    with open(query_num_json, 'r') as file:
        query_num = json.load(file)
    image_ids = list(id_metadata.keys())

    # download_dir = os.path.join(DOWNLOAD_ROOT, f'{query_id:04d}')
    # os.makedirs(download_dir, exist_ok=True)
    download_dir = os.path.join(DOWNLOAD_ROOT)  

    query = CLASS_LABEL[query_id]

    if str(query_id) in query_num:
        downloaded = query_num[str(query_id)]['downloaded']
        duplicate = query_num[str(query_id)]['duplicate']
        small = query_num[str(query_id)]['small']
        page = query_num[str(query_id)]['page_until']
        print(f'Already downloaded {downloaded}, downloading from page {page}...')
    else:
        downloaded = 0
        duplicate = 0
        small = 0
        page = 1
    total_results = -1
    
    trial = 0
    with tqdm(total=total_images) as pbar:
        while downloaded < total_images :
            url = f'https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}'
        
            try:
                response = requests.get(url, headers=headers)
                assert response.status_code == 200
                trial = 0
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                trial += 1
                if response.json()['status'] == 429:
                    print(f'status 429! change api key...', end=' ')
                    api_key_index = (api_key_index+1)%len(api_keys)
                    headers['Authorization'] = api_keys[api_key_index]
                    if api_key_index == 0:
                        print('Sleeping for 5 minutes...')
                        sleep(300)
                else:
                    print(f'Failed to fetch images for {query} ({trial}/5) Retrying after 5 seconds...')
                    sleep(5)

                if trial > 5:
                    print(f'Failed to fetch images for {query} after 5 trials. Skipping...')
                    break
                continue
            
            data = response.json()
            if total_results == -1:
                total_results = data['total_results']
            photos = data['photos']

            for photo in photos:
                photo_id = str(photo['id'])
                if downloaded-duplicate-small >= total_images:
                    print(f'{downloaded-duplicate-small}(={downloaded}-{duplicate}(duplicate)-{small}(small)) images downloaded >= total_images={total_results} for {query}')
                    break
                
                image_path = os.path.join(download_dir, f'{int(photo_id):08d}.jpg')
                w, h = photo['width'], photo['height']
                need_download = True
                if photo_id in image_ids:
                    if not query in id_metadata[photo_id]['queries']:
                        print(f'\tDuplicated ({id_metadata[photo_id]["queries"]})!')
                        duplicate += 1
                        pbar.update(-1)
                    need_download = False

                if need_download:
                    if max(w, h) < 3000:
                        print(f'\tToo small!({w}x{h})')
                        small += 1
                        continue
                    
                    image_url = photo['src']['original']
                    try:
                        image_response = requests.get(image_url, headers=headers)
                        assert image_response.status_code == 200
                    except Exception as e:
                        if isinstance(e, KeyboardInterrupt):
                            raise e
                        print(f'\tFailed to fetch image: {image_url}')
                        continue

                    with open(image_path, 'wb') as f:
                        f.write(image_response.content)

                if photo_id not in id_metadata:
                    id_metadata[photo_id] = {k:v for k, v in photo.items() if k != 'id'}
                    id_metadata[photo_id]['queries'] = [query]
                else:
                    queries = id_metadata[photo_id]['queries']+[query]
                    id_metadata[photo_id]['queries'] = list(set(queries))

                image_ids.append(photo_id)

                pbar.set_description(f'Downloading {query}...(page {page}) {image_path}')
                downloaded += 1
                pbar.update(1)

            if 'next_page' not in data:
                print(f'No more page for {query}!')
                break
            if downloaded >= total_images:
                print(f'{downloaded} images downloaded, given total_images={total_results} for {query}!')
                break

            query_num[str(query_id)] = {
                "total_results": total_results,
                "downloaded": downloaded,
                "duplicate": duplicate,
                "small": small,
                "page_until": page
            }
            with open(query_num_json, 'w') as file:
                json.dump(query_num, file, indent=4)
            with open(id_metadata_json, 'w') as file:
                json.dump(id_metadata, file, indent=4)
            
            page += 1
    
    if total_results == -1:
        return downloaded

    query_num[str(query_id)] = {
        "total_results": total_results,
        "downloaded": downloaded,
        "duplicate": duplicate,
        "small": small,
        "page_until": page
    }
    print(f"Writting metadata to {id_metadata_json} and {query_num_json}...")
    with open(query_num_json, 'w') as file:
        json.dump(query_num, file, indent=4)
    with open(id_metadata_json, 'w') as file:
        json.dump(id_metadata, file, indent=4)
    
    return downloaded

def random_url():
    np.random.seed(42)
    photo_id = np.random.randint(1, 30_000_000)
    return f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg"

if __name__ == '__main__':

    d = 0
    for key, value in list(CLASS_LABEL.items()):
        
        print(f'---- Downloading ({key:03d}) {value} ...----')
        d_ = download_images_by_query(key, 3_000)
        d += d_
        print(f'---- Finished downloading {d_} ({key:03d}) {value} ----\n\n')

        # count_downloaded_images(key)

    # download_image_by_id(28963220)
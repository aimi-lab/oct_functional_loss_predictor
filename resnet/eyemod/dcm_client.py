import requests


class DCMClient():
    def __init__(self, url: str = 'http://130.92.124.127/'):
        self.url = url
        self.endpoint = 'api/dicoms'

    
    def get_dicom(self, params: dict)-> dict:
        url = self.url + self.endpoint
        response = requests.get(url, params=params)

        if response.status_code == 200:
            dicom_images = response.json()
            assert len(dicom_images) == 1, f"Expected a single dicom as response, got {len(dicom_images)}"
            return dicom_images[0]
        else:
            return None
        

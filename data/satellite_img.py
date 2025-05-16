import requests
from PIL import Image
from io import BytesIO

def get_hybrid_image(lat, lon, zoom, api_key, save_path="satellite_image.jpg"):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=800x600&maptype=hybrid&key={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        if img.mode == "P":
            img = img.convert("RGB")
        img.save(save_path)
        print(f"Hybrid image saved to: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
    except IOError as e:
        print(f"Error saving image: {e}")

# Example Usage
api_key = "your key here"
latitude = 51.5080838
longitude = -0.1280996  # Trafalgar Square
zoom_level = 19

get_hybrid_image(latitude, longitude, zoom_level, api_key)

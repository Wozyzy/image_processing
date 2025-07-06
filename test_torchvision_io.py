import torch
from torchvision import io

# Görüntü okuma örneği
def read_image_example():
    try:
        # Görüntüyü oku
        image = io.read_image("your_digit.png")
        print("Görüntü başarıyla okundu!")
        print(f"Görüntü boyutu: {image.shape}")
        return image
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    image = read_image_example() 
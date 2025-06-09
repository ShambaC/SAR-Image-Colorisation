# backend/dummy.py
import sys

if __name__ == "__main__":
    image_path = sys.argv[1]
    latitude = sys.argv[2]
    longitude = sys.argv[3]
    img_date = sys.argv[4]
    output_filename = "jk.jpg"


    print("=== Dummy Colorization Running ===")
    print(f"Received image path: {image_path}")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Date: {img_date}")
    print("Colorized dummy output generated successfully!")
    print(output_filename)

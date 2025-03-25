import os


# Example usage
directory = "/mnt/data1/raiyan/breast_cancer/VLMs-for-Mammograms/vindr/L_CC"  
png_files = list_png_files(directory)

for file in png_files:
    print(file)
# print(png_files)

import os

master_dir = "raiyan/breast_cancer/VLMs-for-Mammograms/evaluated"
subdirs = [d for d in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, d))]
print(subdirs)

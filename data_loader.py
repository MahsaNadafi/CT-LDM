import os
import cv2

import nibabel as nib
'''
in these file we read .hdr ct image
that contain around 50 slice and normalize them
all this single slice can be seved by .png format
'''
dataset_dir = "/hdd3/nadafi/data/original-data/Organized-Normal-CT-Data/"

# Ensure the output directory exists or create it if not
output_dir = "data/CT/single-slice-Normal"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Limit the number of files to read
# num_files_to_read = 5
# files_read = 1
# Iterate over all .hdr files in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.hdr'):
        # Load the Analyze image using nibabel
        img = nib.load(os.path.join(dataset_dir, filename))

        # Get the image data
        img_data = img.get_fdata()

        # Save each slice as a separate PNG file
        for slice_idx  in range(img_data.shape[2]):  # Assuming z-axis is the slice axis
            # Extract the slice data
            slice = img_data[:, :, slice_idx ]
            normalized_img = cv2.normalize(slice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            slice_filename = f'{output_dir}/{filename}_slice_{slice_idx}_normalized.png'
            slice_filename = slice_filename.replace(".hdr", "")
            cv2.imwrite(slice_filename, normalized_img)

            print(f'Saved slice {slice_idx} of {filename} as {slice_filename}')
#
#         # files_read += 1
#         # if files_read >= num_files_to_read:
#         #     break  # Stop reading files once the limit is reached
#
#
# dataset_dir = r"IDM-main/dataset/covid_validation_32_256/hr_256"
# for filename in os.listdir(dataset_dir):
#     if filename.endswith('.png'):
#         image = cv2.imread(os.path.join(dataset_dir, filename), cv2.IMREAD_UNCHANGED)
#         png_filename = f'{dataset_dir}/covid{filename}'
#         cv2.imwrite(png_filename, image)
#         print(f'Saved as {png_filename}')

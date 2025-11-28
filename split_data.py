import os
import shutil

# Set your input directory here
input_dir = r"D:\arshad\Mscthesis\data\Organized COVID19 CT Data\Part 2"
output_base_dir = r"D:\arshad\Mscthesis\data\Organized COVID19 CT Data"  # where you want train/val/test folders

# Create output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base_dir, split), exist_ok=True)

# List and sort the subject files, ignoring folders
all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
subject_ids = sorted(list(set(f.split('_')[0] for f in all_files)), key=lambda x: int(x))

# Sanity check
assert len(subject_ids) == 90, f"Expected 90 subjects, found {len(subject_ids)}"

# Split indices
train_ids = subject_ids[:74]
val_ids = subject_ids[74:75]
test_ids = subject_ids[75:]

# Helper to move files
def move_subject_files(subject_id, split_folder):
    for f in all_files:
        if f.startswith(subject_id + '_'):
            src = os.path.join(input_dir, f)
            dst = os.path.join(output_base_dir, split_folder, f)
            shutil.copy2(src, dst)

# Move the files
for sid in train_ids:
    move_subject_files(sid, 'train')

for sid in val_ids:
    move_subject_files(sid, 'val')

for sid in test_ids:
    move_subject_files(sid, 'test')

print("✅ Done splitting the dataset!")
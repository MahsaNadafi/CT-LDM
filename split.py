import os
import random
from glob import glob
from collections import defaultdict

normal_path = "data/CT/single-slice-Normal"
covid_path = "data/CT/single-slice-COVID19"

out_train = "train.txt"
out_val = "val.txt"
out_test = "test.txt"


# ------------------------------------------------------
# STEP 1 — GROUP IMAGES BY PATIENT ID
# ------------------------------------------------------
def collect_by_patient(folder, class_label):
    images = glob(os.path.join(folder, "*.png"))
    patient_dict = defaultdict(list)

    for img in images:
        name = os.path.basename(img)
        patient_id = name.split("_slice_")[0]
        patient_dict[patient_id].append(name)

    return patient_dict


normal_patients = collect_by_patient(normal_path, "normal")
covid_patients = collect_by_patient(covid_path, "covid")

normal_ids = list(normal_patients.keys())
covid_ids = list(covid_patients.keys())

random.shuffle(normal_ids)
random.shuffle(covid_ids)

print("Total normal patients =", len(normal_ids))
print("Total covid patients =", len(covid_ids))


# ------------------------------------------------------
# STEP 2 — SELECT VALIDATION SET (2 normal + 2 covid)
# ------------------------------------------------------
val_normals = normal_ids[:2]
val_covids = covid_ids[:2]

val_ids = val_normals + val_covids

print("Validation normal patients:", val_normals)
print("Validation covid patients:", val_covids)


# ------------------------------------------------------
# STEP 3 — SPLIT REMAINING PATIENTS INTO TRAIN/TEST
# Keeping class ratio consistent
# ------------------------------------------------------
remaining_normals = normal_ids[2:]
remaining_covids = covid_ids[2:]

def split_class(patients):
    n = len(patients)
    test_size = max(1, int(0.15 * n))  # 15%
    test_set = patients[:test_size]
    train_set = patients[test_size:]
    return train_set, test_set

train_normals, test_normals = split_class(remaining_normals)
train_covids,  test_covids  = split_class(remaining_covids)

train_ids = train_normals + train_covids
test_ids  = test_normals + test_covids

print("Train patients =", len(train_ids))
print("Test patients =", len(test_ids))
print("Val patients =", len(val_ids))


# ------------------------------------------------------
# STEP 4 — WRITE TXT FILES
# ------------------------------------------------------
def write_split(filename, patient_list, patient_dict, covid_dict):
    with open(filename, "w") as f:
        for pid in patient_list:
            if pid in patient_dict:
                imgs = patient_dict[pid]
            else:
                imgs = covid_dict[pid]
            for img in imgs:
                f.write(img + "\n")


# Combine dicts for lookup
all_normal = normal_patients
all_covid  = covid_patients

write_split(out_train, train_ids, all_normal, all_covid)
write_split(out_val, val_ids,     all_normal, all_covid)
write_split(out_test, test_ids,   all_normal, all_covid)

print("train.txt, val.txt, test.txt generated.")

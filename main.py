# coding: utf-8

import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation
import warnings

warnings.filterwarnings("ignore")

# Load training images
images_train = []
for img_path in sorted(glob.glob("CASIA_Iris_Image_Dataset/training/*.png")):
    img = cv2.imread(img_path)
    if img is None:
        print(
            f"Warning: Could not load image {img_path}. It may be corrupted or in an unsupported format."
        )
    else:
        images_train.append(img)

# Load testing images
images_test = []
for img_path in sorted(glob.glob("CASIA_Iris_Image_Dataset/testing/*.png")):
    img = cv2.imread(img_path)
    if img is None:
        print(
            f"Warning: Could not load image {img_path}. It may be corrupted or in an unsupported format."
        )
    else:
        images_test.append(img)

print("Training images loaded:", len(images_train))
print("Testing images loaded:", len(images_test))


"""TRAINING"""

# running Localization, Normalization, Enhancement and Feature Extraction on all the training images
boundary_train, centers_train = IrisLocalization(images_train)
normalized_train = IrisNormalization(boundary_train, centers_train)
enhanced_train = ImageEnhancement(normalized_train)
feature_vector_train = FeatureExtraction(enhanced_train)
print("Training data processed.")


"""TESTING"""

# running Localization, Normalization, Enhancement and Feature Extraction on all the testing images
boundary_test, centers_test = IrisLocalization(images_test)
normalized_test = IrisNormalization(boundary_test, centers_test)
enhanced_test = ImageEnhancement(normalized_test)
feature_vector_test = FeatureExtraction(enhanced_test)
print("Testing data processed.")


# Initialize lists for CRR and matching results
crr_L1 = []
crr_L2 = []
crr_cosine = []
match_cosine = []
match_cosine_ROC = []

# Performing Matching and CRR scores for 10,40,60,80,90,107 number of dimensions in the reduced feature vector
components = [10, 40, 60, 80, 90, 107]

print("Begin Matching test data with the train data")
for comp in components:
    # Running matching for all the dimensions specified in "components"
    comp_match_L1, comp_match_L2, comp_match_cosine, comp_match_cosine_ROC = (
        IrisMatching(feature_vector_train, feature_vector_test, comp, 0)
    )

    # Calculating CRR for all the dimensions specified in "components"
    comp_crr_L1, comp_crr_L2, comp_crr_cosine = PerformanceEvaluation(
        comp_match_L1, comp_match_L2, comp_match_cosine
    )

    # combining the results of all the dimensional feature vector into one array
    crr_L1.append(comp_crr_L1)
    crr_L2.append(comp_crr_L2)
    crr_cosine.append(comp_crr_cosine)
    match_cosine.append(comp_match_cosine)
    match_cosine_ROC.append(comp_match_cosine_ROC)


# Performing Matching and calculating CRR score for the original feature vector (without dimensionality reduction)
orig_match_L1, orig_match_L2, orig_match_cosine, orig_match_cosine_ROC = IrisMatching(
    feature_vector_train, feature_vector_test, 0, 1
)
orig_crr_L1, orig_crr_L2, orig_crr_cosine = PerformanceEvaluation(
    orig_match_L1, orig_match_L2, orig_match_cosine
)
print("Completed Matching")


# Table for CRR rates for the original and reduced feature set(components=107)
print("\n\n")
table = pd.DataFrame(
    {
        "Similarity Measure": ["L1", "L2", "Cosine Distance"],
        "CRR for Original Feature Set": [orig_crr_L1, orig_crr_L2, orig_crr_cosine],
        "CRR for Reduced Feature Set (107)": [crr_L1[5], crr_L2[5], crr_cosine[5]],
    }
)
print("Recognition results using Different Similarity Measures:\n")
print(table.iloc[0], "\n")
print(table.iloc[1], "\n")
print(table.iloc[2])

# Plotting the incresing CRR for cosine similarity with the incresing dimensionality
plt.plot(components, crr_cosine, marker="o")
plt.axis([10, 107, 0, 100])
plt.ylabel("Correct Recognition Rate (Cosine)")
plt.xlabel("Dimensionality of the feature vector")
plt.title("Recognition Results")
plt.show()


# After computing match_cosine_ROC, calculate FMR and FNMR
fmr_all = []
fnmr_all = []
thresh = [0.4, 0.5, 0.6]

for q in range(len(thresh)):  # Loop over each threshold
    false_accept = 0
    false_reject = 0
    num_1 = len(
        [i for i in match_cosine_ROC[5][q] if i == 1]
    )  # Count of accepted images
    num_0 = len(
        [i for i in match_cosine_ROC[5][q] if i == 0]
    )  # Count of rejected images

    for p in range(len(match_cosine[5])):
        if match_cosine[5][p] == 0 and match_cosine_ROC[5][q][p] == 1:
            false_accept += 1  # Incorrectly accepted (False Match)
        if match_cosine[5][p] == 1 and match_cosine_ROC[5][q][p] == 0:
            false_reject += 1  # Incorrectly rejected (False Non-Match)

    # Calculate FMR and FNMR
    if num_1 > 0:
        fmr = false_accept / num_1  # False matches / total accepted
    else:
        fmr = 0

    if num_0 > 0:
        fnmr = false_reject / num_0  # False non-matches / total rejected
    else:
        fnmr = 0

    fmr_all.append(fmr)
    fnmr_all.append(fnmr)

# Print FMR and FNMR results
print("\n\n")
list = pd.DataFrame(
    {
        "Threshold": thresh,
        "FMR": fmr_all,
        "FNMR": fnmr_all,
    }
)
print("ROC Measures:\n")
print(list.iloc[0], "\n")
print(list.iloc[1], "\n")
print(list.iloc[2])

# Plotting the ROC Curve (FMR vs FNMR)
plt.plot(fmr_all, fnmr_all, marker="o")
plt.title("ROC Curve")
plt.xlabel("False Match Rate (FMR)")
plt.ylabel("False Non-Match Rate (FNMR)")
plt.show()

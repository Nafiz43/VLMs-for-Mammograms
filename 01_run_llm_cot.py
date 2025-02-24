"""
Consider yourself a radiologist analyzing a mammogram image. I will provide you with a mammogram image. Your task is to analyze the image and extract key diagnostic information, including breast composition, BIRADS category, and any significant findings. Present the output in a structured JSON format with the following keys: IMG_ID, Breast_Composition, BIRADS, and Findings. Ensure the response is precise, medically relevant, and well-organized.

Step 1: First find out the Breast Density Category in ACR Format where 
- ACR A (Almost entirely fatty): The breast is composed mostly of fat with minimal glandular tissue. Low density → Easier to detect abnormalities. Least associated with an increased risk of breast cancer.

- ACR B (Scattered areas of fibroglandular density): Mostly fatty, but with some scattered dense tissue. Abnormalities are still generally well-detected. Slightly increased breast cancer risk compared to ACR A.


- ACR C (Heterogeneously dense): A significant amount of glandular tissue, making the breast more dense. Can obscure small tumors, making detection more challenging. Moderately increased risk of breast cancer.


- ACR D (Extremely dense): The breast is almost entirely composed of dense fibroglandular tissue. High density makes it very difficult to detect abnormalities on a mammogram. Significantly increased risk of breast cancer.


Step 2: Then determine any abnormal findings (or tumors) in the image. Findings are abnormalities or observations detected in a mammogram. Each type indicates different levels of concern: 

- CALC (Calcification): Tiny calcium deposits in the breast tissue. Can be benign (due to aging, injury, or inflammation) or suspicious (clustered, irregular shapes, which may indicate early cancer). Further evaluation is required for suspicious calcifications.


- CIRC (Well-defined/Circumscribed Masses): Round, smooth masses with clear borders.
Often benign, such as cysts or fibroadenomas.
Requires further imaging or biopsy if growth or suspicious features are noted.


- SPIC (Spiculated Masses): Irregular masses with spiky, radiating edges. Highly suspicious for malignancy (invasive cancer). Biopsy is usually recommended for confirmation.


- MISC (Other Ill-Defined Masses): Masses that do not fit into other well-defined categories. Can be benign or malignant, requiring further evaluation with additional imaging or biopsy.


- ARCH (Architectural Distortion): Distortion of normal breast tissue structure without a clearly defined mass. Can be caused by prior surgery, trauma, or malignancy. Further imaging (MRI, ultrasound) or biopsy is often needed.


- ASYM (Asymmetry): One breast appears different from the other in density or structure. Can be due to normal variations, prior surgery, or an underlying lesion. If new or developing asymmetry is observed, additional tests may be needed.


- NORM (Normal): No suspicious findings. Routine screening continues as per guidelines.


Step 3: Now finally, determine the BIRADS of the mammogram where 


 "BIRADS": "<BIRADS category; any values between 0 to 6. BI-RADS category is a standardized classification for breast imaging findings, ranging from 0 to 6, where: BIRADS 0 means the test is incomplete or missing data. BI-RADS 1 indicates a negative result with no abnormalities; BI-RADS 2 signifies benign findings with no suspicion of cancer; BI-RADS 3 suggests a benign lesion, requiring short-term follow-up to confirm stability; BI-RADS 4 represents a suspicious abnormality needing biopsy, further divided into 4A (low suspicion), 4B (moderate suspicion), and 4C (high suspicion); BI-RADS 5 is highly suggestive of malignancy with a high probability of cancer; and BI-RADS 6 confirms a known malignancy with a biopsy-proven cancer diagnosis.",


Step 4: Please follow the below given JSON format for your response:
{
    "IMG-ID": "<Image_Filename>",
    "BREAST-COMPOSITION": "<Description of breast tissue composition>",
    "BIRADS": "<BIRADS category>",
    "FINDINGS": "<Summary of any abnormalities, calcifications, or other observations>"
}
"""
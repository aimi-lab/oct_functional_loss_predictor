import pydicom


def get_dicom_tag(dataset: pydicom.FileDataset, tag_name: str):
    """
    Recursively search for a DICOM tag by name in a dataset and its nested sequences.
    """
    for elem in dataset:
        if elem.name == tag_name:
            return elem.value  # Return the value if found
        
        # If the element is a sequence, search inside each item
        if elem.VR == "SQ":  # VR (Value Representation) "SQ" means it's a sequence
            for item in elem.value:
                result = get_dicom_tag(item, tag_name)
                if result is not None:
                    return result  # Return the first match found
    
    return None

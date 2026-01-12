from pathlib import Path

INPUT_BASE = Path('in/')
DATA_BASE_PATH = Path('in/data')

RAW_BASE = DATA_BASE_PATH / 'raw'

EYESUITE_RAW = RAW_BASE / '000_eyesuite_export.csv'
GPATTERN_ORDER = RAW_BASE / 'gpattern_x_y_coordinates.xlsx'
TRABEC_RAW = RAW_BASE / 'glaucoma_trabeculectomy_clinical_data.xlsx'
SCLEREC_RAW = RAW_BASE / 'glaucoma_deep_sclerectomy_clinical_data.xlsx'

PATIENT_LINK = RAW_BASE / 'overview_patients_anon.xlsx'
DICOM_LINK = RAW_BASE / 'overview_dicoms_anon.xlsx'

IOP_DATASET_BASE = DATA_BASE_PATH / '2024_IOP_prediction'
IOP_FORMATTED_BASE = IOP_DATASET_BASE / 'formatted'

OCT2VF_DATASET_BASE = DATA_BASE_PATH / '2024_OCT2VF_processed'


IMAGE_BASE = Path('/Users/moritzschmid/Datasets/2024_IOP_prediction/images')

DISCOVERY_MODELS = INPUT_BASE / 'discovery_models'

OUTPUT_BASE = Path('out/')
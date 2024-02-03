import os
from pathlib import Path

ORIGINAL_DATASET_PATH = Path('data/celeba/Eyeglasses')
NEG_POS_FILENAMES_PATH = Path('FINAL_REPO/generate_jobfiles/diversity_eyeglasses/Eyeglasses_negative_Male_positve.txt')
POS_POS_FILENAMES_PATH = Path('FINAL_REPO/generate_jobfiles/diversity_eyeglasses/Eyeglasses_positive_Male_positve.txt')


neg_filenames = os.listdir(os.path.join(ORIGINAL_DATASET_PATH, 'negative'))
pos_filenames = os.listdir(os.path.join(ORIGINAL_DATASET_PATH, 'positive'))


with open(NEG_POS_FILENAMES_PATH, 'r') as f:
    pos_neg_filenames = f.read().splitlines()
with open(POS_POS_FILENAMES_PATH, 'r') as f:
    pos_pos_filenames = f.read().splitlines()
neg_neg_filenames = [filename for filename in neg_filenames if filename not in pos_neg_filenames and '.jpg' in filename]

non_diverse_path = Path(str(ORIGINAL_DATASET_PATH).rpartition('/')[0] + '/Eyeglasses_non_diverse')
only_male_path = Path(str(ORIGINAL_DATASET_PATH).rpartition('/')[0] + '/Eyeglasses_Male_only_positive')

(non_diverse_path / 'negative').mkdir(parents=True, exist_ok=True)
(non_diverse_path / 'positive').mkdir(parents=True, exist_ok=True)
(only_male_path / 'negative').mkdir(parents=True, exist_ok=True)
(only_male_path / 'positive').mkdir(parents=True, exist_ok=True)

for filename in neg_neg_filenames:
    os.link(os.path.join(ORIGINAL_DATASET_PATH, 'negative', filename), os.path.join(non_diverse_path, 'negative', filename))
for filename in pos_neg_filenames:
    os.link(os.path.join(ORIGINAL_DATASET_PATH, 'negative', filename), os.path.join(only_male_path, 'negative', filename))
for filename in pos_pos_filenames:
    os.link(ORIGINAL_DATASET_PATH / 'positive' / filename, non_diverse_path / 'positive' / filename)
    os.link(ORIGINAL_DATASET_PATH / 'positive' / filename, only_male_path / 'positive' / filename)
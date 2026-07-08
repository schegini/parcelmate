import re

CONNECTIVITY_NAME = 'connectivity'
PARCELLATION_NAME = 'parcellation'
SUBNETWORK_NAME = 'subnetwork'
KNOCKOUT_NAME = 'knockout'
STABILITY_NAME = 'stability'
SAMPLE_NAME = 'sample'
LOSS_NAME = 'loss'
MEAN_ACTIVATION_NAME = 'mean_activations'
BASELINE_NAME = 'baseline'
HEALTHY_NAME = 'healthy'

N_SAMPLES = 4
N_TOKENS = 100000
EXTENSION = '.h5'
INPUT_NAME_RE = re.compile(r'(%s|%s)_(.+)_(%s\d+|avg)%s' % (
    CONNECTIVITY_NAME, PARCELLATION_NAME, SAMPLE_NAME, EXTENSION)
)

OUTPUT_DIR = 'results'
PLOT_DIR = 'plots'

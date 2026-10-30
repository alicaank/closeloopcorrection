"""VIPE is the default evaluator; historical VGGT implementation preserved separately."""
from score_pair_vggt_legacy import transform
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vipe_eval_20261030'))
from score import main
if __name__=='__main__':main('pair')

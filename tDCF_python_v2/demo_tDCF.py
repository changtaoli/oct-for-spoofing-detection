# example of use of the evaluate_tDCF_asvspoof19 function using
# the ASVspoof 2019 official baseline contermeasure systems.
# (score files stored in folder "scores")

from evaluate_tDCF_asvspoof19 import evaluate_tDCF_asvspoof19

# set here the track and baseline system to use
track = 'LA';   # 'LA' or 'PA'
system = 'B01'; # 'B01' or 'B02'

ASV_SCOREFILE = 'scores/ASVspoof2019_' + track + '_eval_asv_scores.txt'
CM_SCOREFILE = 'scores/' + system + '_' + track + '_primary_eval.txt'

# set the 3rd argument to true to use the legacy t-DCF formulation
# used in the ASVspoof 2019 challenge (discouraged)
evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE, False);   

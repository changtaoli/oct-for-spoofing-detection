A Python code package for computing t-DCF and EER metrics for ASVspoof2019.
(Version 2.0)

The main difference with regard to Version 1.0 is the use of a revisited formulation
of the tandem detection cost function. Further details can be found in the paper:

Tomi Kinnunen, Héctor Delgado, Nicholas Evans,Kong Aik Lee, Ville Vestman, 
Andreas Nautsch, Massimiliano Todisco, Xin Wang, Md Sahidullah, Junichi Yamagishi, 
and Douglas A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
Audio, Speech and Language Processing (TASLP).

USAGE:

 python evaluate_tDCF_asvspoof19.py <CM_SCOREFILE> <ASV_SCOREFILE> <LEGACY>

where 

  CM_SCOREFILE points to your own countermeasure score file.
  ASV_SCOREFILE points to the organizers' ASV scores.
  LEGACY is a boolean flag. If set to True, the t-DCF formulation
        employed in the ASVspoof 2019 challenge is used (discouraged).

NOTE! There are two different ASV score files provided by the organizers:
        One for the physical access (PA) scenario and the other for the logical access (LA) scenario.
        Be sure to point to the right ASV score file according to the scenario (LA or PA).

A demo script "demo_tDCF.py" is provided. It computes the normalized minimum t-DCF
   of the baseline systems for the LA and PA scenarios.

REQUIREMENTS:
 -python3 (tested with version 3.6.5)
 -numpy (tested with version 1.15.4)
 -matplotlib (tested with version 2.2.2)

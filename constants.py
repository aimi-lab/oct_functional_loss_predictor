# BAD_DICOMS = [
#     '0.79538795621850920200318210048437.dcm', # off-centered wrt fovea, 61 slices
#     '0.085497772826579292020052915454429.dcm', # off-centered, at the limit of acceptability

#     '0.600715769734567201912202223011.dcm', # 768*49*496, grid not included

#     '0.587302059674310520191129101912160.dcm', # bad segmentation, missing slices partially
#     '0.5111042496334316202008080310521.dcm', # bad segmentation, missing slices partially
#     '0.24859855475304582202001102344425.dcm', # bad segmentation on GCL+IPL

#     '0.00901506934734763202006052247582.dcm', # Graber, bad eye condition
#     '0.0631088661323808420200318214751385.dcm', # bad scan

#     '0.4154344421883274201912062145341.dcm', # 31-slice OCT, badly located
#     '0.41949626031308262019120621442313.dcm', # 31-slice OCT, duplicate, same patient as previous

#     '0.8775985789846622202001222056243.dcm', # duplicate, discarded the worst
#     '0.8768404153533468202001172156489.dcm', # duplicate, discarded the worst
#     '0.17791439368292428201908092336350.dcm', # duplicate, discarded the worst
#     '0.36860274028433615202007102125491.dcm', # duplicate, discarded the worst
#     '0.7261601829557495201911012248261.dcm', # duplicate, discarded the worst 
#     '0.16751833686955195201911020411192.dcm', # duplicate, discarded the worst
#     '0.9622443541708608202005300319321.dcm', # duplicate, discarded the worst
#     '0.15682706942634056201912202223411.dcm', # duplicate, discarded the worst
#     '0.34696910639618017201908240255447.dcm', # duplicate, discarded the worst
#     '0.90340671730386420200318223201318.dcm', # duplicate, discarded the worst, 46 slices btw
# ]

# region
# 0.9139383369655992202001250443384.dcm think about it, at the limit of acceptability for being off-centered

# 0.27042486065552795201912202223055.dcm is slightly off-centered, but kept
# 0.8287657829135497202001180229203.dcm is slightly off-centered, but kept
# 0.02351161279879119820200814114513265.dcm is slightly off-centered, but kept
# 0.4353568374343947202007180233011.dcm has some minor part of slice missing, but accepted

# 0.771767945388223920200807111320157.dcm slightly bad segmentation but kept

# Buri has two eyes, but one was discarded as badly segmented 0.587302059674310520191129101912160.dcm
# Elsener & Glauser have two eyes, both 49 slices
# endregion

RETINAL_LAYERS = ['RNFL', 'GCL+IPL', 'INL+OPL', 'ONL', 'PR+RPE', 'CC+CS', 'SRF', 'PED', 'RT', 'BACKGROUND']

FEATURES = ["THICKNESS_S6", 
            "THICKNESS_N6", 
            "THICKNESS_I6", 
            "THICKNESS_T6", 
            "THICKNESS_S3", 
            "THICKNESS_N3", 
            "THICKNESS_I3", 
            "THICKNESS_T3", 
            "THICKNESS_C1", 
            "THICKNESS_BG",
            "VOLUME_S6", 
            "VOLUME_N6", 
            "VOLUME_I6", 
            "VOLUME_T6", 
            "VOLUME_S3", 
            "VOLUME_N3", 
            "VOLUME_I3", 
            "VOLUME_T3", 
            "VOLUME_C1",
            "VOLUME_BG"
            ]

RNDM_STATE = 55
CV = 5
ERROR_ABBR = {'absolute_error': 'MAE', 'squared_error': 'MSE'}

G_POINTS = 59
G_CLUSTERS = {
    (0, 0): None,

    # 1 
    (20, 20): 'Cluster 1',
    (-20, 20): 'Cluster 1',
    (-40, 40): 'Cluster 1',
    (-80, 20): 'Cluster 1',

    # 2
    (40, 40): 'Cluster 2',
    (80, 20): 'Cluster 2',
    (260, 40): 'Cluster 2',
    (140, 40): 'Cluster 2',
    (200, 40): 'Cluster 2',
    (-20, 80): 'Cluster 2',
    (20, 80): 'Cluster 2',

    # 3
    (-80, 80): 'Cluster 3',
    (80, 80): 'Cluster 3',
    (120, 120): 'Cluster 3',
    (200, 120): 'Cluster 3',
    (40, 140): 'Cluster 3',
    (-40, 140): 'Cluster 3',

    # 4
    (-120, 120): 'Cluster 4',
    (-120, 200): 'Cluster 4',
    (-40, 200): 'Cluster 4',
    (40, 200): 'Cluster 4',
    (120, 200): 'Cluster 4',
    (200, 200): 'Cluster 4',
    (80, 260): 'Cluster 4',
    (-80, 260): 'Cluster 4',

    # 5
    (-220, 40): 'Cluster 5',
    (-260, 80): 'Cluster 5',
    (-200, 200): 'Cluster 5',
    (-200, 120): 'Cluster 5',

    # 6
    (-220, -40): 'Cluster 6',
    (-260, -80): 'Cluster 6',
    (-200, -200): 'Cluster 6',
    (-200, -120): 'Cluster 6',

    # 7
    (-120, -120): 'Cluster 7',
    (-120, -200): 'Cluster 7',
    (-40, -200): 'Cluster 7',
    (40, -200): 'Cluster 7',
    (120, -200): 'Cluster 7',
    (200, -200): 'Cluster 7',
    (80, -260): 'Cluster 7',
    (-80, -260): 'Cluster 7',

    # 8
    (80, -80): 'Cluster 8',
    (200, -120): 'Cluster 8',
    (40, -140): 'Cluster 8',
    (-40, -140): 'Cluster 8',
    (120, -120): 'Cluster 8',

    # 9
    (80, -20): 'Cluster 9',
    (-30, -90): 'Cluster 9',
    (30, -90): 'Cluster 9',
    (-80, -80): 'Cluster 9',
    (80, -80): 'Cluster 9',
    (40, -40): 'Cluster 9',
    (140, -40): 'Cluster 9',
    (200, -40): 'Cluster 9',
    (260, -40): 'Cluster 9',

    # 10
    (20, -20): 'Cluster 10',
    (-40, -40): 'Cluster 10',
    (-80, -20): 'Cluster 10',
    (-20, -20): 'Cluster 10',
}

assert len(G_CLUSTERS.keys()) == len(set(G_CLUSTERS.keys()))
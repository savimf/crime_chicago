BRZ_PATH = '../../bronze'
SLV_PATH = '../../silver'
GLD_PATH = '../../gold'
MODEL_PATH = '../../models'
IMG_PATH = '../../images'

COLORS = {
    'THFT': 'dimgrey',
    'ASLT': 'indianred'
}

GENERAL_THFT = [
    'THEFT', 'BURGLARY', 'MOTOR VEHICLE THEFT',
    'ROBBERY', 'CRIMINAL TRESPASS', 'AUTO THEFT',
    'DECEPTIVE PRACTICE', 'LIQUOR LAW VIOLATION',
    'PROSTITUTION'
]

GENERAL_ASLT = [
    'BATTERY', 'ASSAULT', 'WEAPONS VIOLATION', 'CRIMINAL DAMAGE',
    'CRIMINAL SEXUAL ASSAULT', 'HOMICIDE', 'SEX OFFENSE',
    'KIDNAPPING', 'INTIMIDATION', 'OBSCENITY', 'ARSON',
    'STALKING', 'HUMAN TRAFFICKING',
    'NARCOTICS', 'OFFENSE INVOLVING CHILDREN',
    'PUBLIC PEACE VIOLATION', 'INTERFERENCE WITH PUBLIC OFFICER',
    'CONCEALED CARRY LICENSE VIOLATION', 'PUBLIC INDECENCY',
    'OTHER NARCOTIC VIOLATION', 'GAMBLING'
]

CENSUS_DATA = [
    'pct_housing_crowded', 'pct_hh_below_pvty',
    'pct_16_unemployed', 'pct_25_without_hsd',
    'pct_dependents', 'per_capita_income'
]

cluster_mapping = {
    'ASLT': {
        0: '2',
        1: '3',
        2: '0',
        3: '1'
    },
    'THFT': {
        0: '3',
        1: '2',
        2: '1',
        3: '0'
    }
}

CLUSTER_COLORS = {
    'rgb': {
        '0': 'rgb(252, 187, 161)',
        '1': 'rgb(252, 146, 114)',
        '2': 'rgb(251, 106, 74)',
        '3': 'rgb(165, 15, 21)'
    },
    'hex': {
        '0': '#fcbba1',
        '1': '#fc9272',
        '2': '#fb6a4a',
        '3': '#a50f15',
    }
}

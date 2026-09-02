'''
Eczema binary label definitions.

"Eczema" and "dermatitis" are synonyms in modern dermatology for one family of inflammatory
skin disease, defined histologically by spongiosis (intercellular epidermal edema) and
clinically by itch, poorly-defined margins, and vesiculation/oozing acutely giving way to
lichenification/excoriation chronically. ICD-11 groups them as EA80-EA8Z "Dermatitis and
eczema".

Scope adopted here is the *core spongiotic* subset of that block:
    atopic, contact (allergic + irritant), dyshidrotic/pompholyx, nummular/discoid,
    asteatotic/xerotic, pityriasis alba.

Deliberately excluded as distinct entities despite living in the ICD-11 dermatitis block:
    seborrheic dermatitis (Malassezia-driven), stasis dermatitis (venous hypertension),
    lichen simplex chronicus / neurodermatitis / prurigo nodularis (scratch-driven
    lichenification), acrodermatitis enteropathica (zinc deficiency).

Classes are enumerated exactly rather than keyword-matched, because substring matching on
"eczema"/"dermatitis" is wrong in both directions:

  Named "dermatitis" but NOT eczema:
    perioral dermatitis (rosacea-family papulopustular), dermatitis herpetiformis
    (autoimmune blistering, celiac-associated), factitial dermatitis (self-inflicted),
    radiodermatitis (radiation injury), dermatomyositis (connective tissue disease),
    steroid-use dermatitis (iatrogenic).

  Eczema without saying so:
    pompholyx, xerotic/asteatotic eczema, erythema craquele, pityriasis alba.

  Eczema mimickers that must stay negative:
    mycosis fungoides / cutaneous T-cell lymphoma masquerades as recalcitrant eczema for
    years but is a malignancy; tinea, psoriasis, lichen planus and scabies are the standard
    clinical differentials. Seborrheic keratosis shares a word with seborrheic dermatitis and
    is otherwise unrelated.
'''

# Dermnet ships 23 coarse buckets rather than individual diagnoses.
# Poison ivy is rhus dermatitis, i.e. textbook allergic contact dermatitis.
DERMNET_ECZEMA_CLASSES = frozenset([
    'Eczema Photos',
    'Atopic Dermatitis Photos',
    'Poison Ivy Photos and other Contact Dermatitis',
])

FITZPATRICK17K_ECZEMA_CLASSES = frozenset([
    'eczema',
    'dyshidrotic eczema',
    'allergic contact dermatitis',
])

# Infantile_Atopic_Dermatitis and Erythema_Craquele (asteatotic eczema) were both labeled
# negative by the previous keyword match purely because the string lacked "eczema".
# Xerosis stays negative: dry skin without inflammation, distinct from Dry_Skin_Eczema.
SD198_ECZEMA_CLASSES = frozenset([
    'Eczema',
    'Acute_Eczema',
    'Nummular_Eczema',
    'Dyshidrosiform_Eczema',
    'Dry_Skin_Eczema',
    'Erythema_Craquele',
    'Infantile_Atopic_Dermatitis',
    'Allergic_Contact_Dermatitis',
    'Pityriasis_Alba',
])

ECZEMA_CLASSES = {
    'dermnet': DERMNET_ECZEMA_CLASSES,
    'fitzpatrick17k': FITZPATRICK17K_ECZEMA_CLASSES,
    'sd198': SD198_ECZEMA_CLASSES,
}


def is_eczema_target(dataset, class_name):
    '''
    Returns whether a dataset's original class name is core spongiotic eczema

    Args:
        dataset : str
            One of dermnet, fitzpatrick17k, sd198
        class_name : str
            Original class name as it appears in that dataset
    Returns:
        bool : True if the class is eczema
    '''

    classes = ECZEMA_CLASSES[dataset]

    # fitzpatrick17k labels come from a CSV free-text column; the other two are directory names
    if dataset == 'fitzpatrick17k':
        return class_name.strip().lower() in classes

    return class_name in classes

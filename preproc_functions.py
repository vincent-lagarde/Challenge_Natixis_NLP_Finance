
def suppr_footnotes(text):
    '''
    Description : fonction pour supprimer les footnotes d'un texte (références biblio, etc)  (à vérifier si permet de supprimer toutes les footnotes)
    - input : colonne 'text'
    - output : colonne 'text' modifiée afin de supprimer les footnotes / citations

    '''
    
    txt = text

    try:
        
        txt_list = txt.split('References', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    try:
        txt_list = txt.split('Footnotes', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    try:
        txt_list = txt.split('      [1]', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    try:
        txt_list = txt.split('See also footnotes', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass

    try:
        txt_list = txt.split(' References ', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass

    try:
        txt_list = txt.split(' 1. ', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    try:
        txt_list = txt.split('SEE ALSO', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    try:
        txt_list = txt.split('See also', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    try:
        txt_list = txt.split('Thank you. ', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass

    try:
        txt_list = txt.split('Thank you for your attention.  ', maxsplit=1)
        if len(txt)>1:
            txt = txt_list[0]
        else:
            1/0
    except:
        pass
    
    return txt
    
    